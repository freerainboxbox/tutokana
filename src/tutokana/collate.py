"""Batching: audio, padding, register positions and per-head targets.

Register positions are found by scanning `input_ids` for the register ids and zipping them
against the slots the renderer declared; a mismatch raises immediately rather than
mislabelling silently.

Language-model cross-entropy is masked off register positions *and* audio positions. Audio
masking is not optional: supervising the constant `<|audio|>` id collapses the
representations at exactly the positions carrying speech, producing a model that cannot hear.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import SAMPLE_RATE, TargetStats, Utterance
from .prompting import TargetSpec, render_prompt, render_target
from .tokens import UNTRAINED_FIELDS, register_tokens

IGNORE_INDEX = -100
#: `<channel|>` — closes the empty thought block the chat template appends, and therefore
#: marks the exact boundary between the eval prompt and the supervised completion.
END_OF_CHANNEL_TOKEN = "<channel|>"


def head_key(level: str, field: str) -> str:
    return f"{level}.{field}"


@dataclass
class HeadBatch:
    """Flattened supervision for one (level, field) head across the whole batch.

    `positions` is (M, 2) of (batch index, sequence index). `native` is the field's reported
    label, which is what every metric compares against; `target` is that z-scored, which is
    what the correlation term consumes. `raw` is whatever the head's pointwise loss reads,
    and equals `native` for every head except `binomial`, where it is the annotator vote
    count. `phone_id` is the FiLM conditioning index, present only for phone-level heads.

    `word_index` names the word each row belongs to, batch-unique and present only for
    word-level heads. It exists so a head can be handed a sibling head's output for the same
    word: the three word registers are emitted together and therefore land in the same order
    in every word-level batch, but that is an accident of `render_target`'s slot order and
    nothing enforced it. Any consumer must check the indices match rather than assume it.
    """

    positions: torch.Tensor
    target: torch.Tensor
    raw: torch.Tensor
    native: torch.Tensor
    phone_id: torch.Tensor | None = None
    word_index: torch.Tensor | None = None

    def to(self, device) -> "HeadBatch":
        return HeadBatch(
            positions=self.positions.to(device),
            target=self.target.to(device),
            raw=self.raw.to(device),
            native=self.native.to(device),
            phone_id=None if self.phone_id is None else self.phone_id.to(device),
            word_index=None if self.word_index is None else self.word_index.to(device),
        )

    def __len__(self) -> int:
        return self.positions.shape[0]


class Collator:
    """Turns a list of `Utterance` into a model-ready batch.

    `phone_vocab` sizes the FiLM conditioning table; symbols outside it map to the shared
    fallback row at index 0, so an unseen phone degrades gracefully instead of crashing.
    """

    def __init__(
        self,
        processor,
        spec: TargetSpec,
        stats: TargetStats,
        phone_vocab: tuple[str, ...],
        max_length: int | None = 2048,
        with_target: bool = True,
    ):
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.spec = spec
        self.stats = stats
        self.max_length = max_length
        # The scored evaluation also renders the target — it teacher-forces the canonical
        # phones so registers align one-to-one with gold. Only the generative-mode decode
        # sets with_target=False.
        self.with_target = with_target
        self.register_ids = register_tokens(self.tokenizer)
        self.id_to_register = {v: k for k, v in self.register_ids.items()}
        self.phone_to_id = {p: i + 1 for i, p in enumerate(phone_vocab)}
        self.n_phone_ids = len(phone_vocab) + 1
        eoc = self.tokenizer.encode(END_OF_CHANNEL_TOKEN, add_special_tokens=False)
        if len(eoc) != 1:
            raise RuntimeError(f"{END_OF_CHANNEL_TOKEN} is not atomic: {eoc}")
        self.end_of_channel_id = eoc[0]

    # -- targets ---------------------------------------------------------------------

    def _native_value(self, utterance: Utterance, slot) -> float:
        """The label on the field's reported scale — what the metrics compare against."""
        if slot.level == "utterance":
            return utterance.utterance_targets()[slot.field]
        word = utterance.words[slot.word]
        if slot.level == "word":
            return getattr(word, slot.field)
        return word.phone_accuracy[slot.phone]

    def _raw_value(self, utterance: Utterance, slot) -> float:
        """The label the head's pointwise loss reads.

        Identical to the reported label everywhere except word stress, where the released
        5/10 score is the median of five annotators and the head is trained on how many of
        them agreed. Reporting is unaffected: the head maps its prediction back onto the
        published median scale.
        """
        if (slot.level, slot.field) == ("word", "stress"):
            return float(utterance.words[slot.word].stress_votes)
        return self._native_value(utterance, slot)

    def _phone_id(self, utterance: Utterance, slot) -> int:
        return self.phone_to_id.get(utterance.words[slot.word].phones[slot.phone], 0)

    # -- batching --------------------------------------------------------------------

    def __call__(self, utterances: list[Utterance]) -> dict:
        rendered = [render_target(u, self.spec) for u in utterances]
        texts = [
            render_prompt(self.processor, u.text) + (r.text if self.with_target else "")
            for u, r in zip(utterances, rendered)
        ]
        audio = [np.asarray(u.audio, dtype=np.float32) for u in utterances]

        # Right padding during training keeps absolute positions aligned with the target;
        # generation flips to left padding so every sequence ends at the same index.
        previous_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right" if self.with_target else "left"
        try:
            batch = self.processor(
                text=texts,
                audio=audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
        finally:
            self.tokenizer.padding_side = previous_side

        input_ids = batch["input_ids"]
        seq_len = input_ids.shape[1]
        if self.max_length is not None and seq_len > self.max_length:
            # Refuse rather than truncate: a truncated target silently drops registers and
            # would train the heads against the wrong scores.
            longest = max(utterances, key=lambda u: u.duration_s)
            raise ValueError(
                f"batch is {seq_len} tokens, over max_length={self.max_length} "
                f"(longest utterance {longest.duration_s:.1f}s, text {longest.text[:60]!r})"
            )

        out = dict(batch)
        if not self.with_target:
            out["utterances"] = utterances
            return out

        out["labels"] = self._language_labels(input_ids, batch.get("attention_mask"))
        out["heads"] = self._head_batches(input_ids, utterances, rendered)
        out["utterances"] = utterances
        return out

    def _language_labels(self, input_ids: torch.Tensor, attention_mask) -> torch.Tensor:
        """Cross-entropy targets over the completion text only."""
        labels = input_ids.clone()
        register_id_tensor = torch.tensor(
            sorted(self.register_ids.values()), dtype=input_ids.dtype
        )
        for b in range(input_ids.shape[0]):
            boundary = (input_ids[b] == self.end_of_channel_id).nonzero(as_tuple=True)[0]
            if boundary.numel() != 1:
                raise RuntimeError(
                    f"expected exactly one {END_OF_CHANNEL_TOKEN} in the rendered text, "
                    f"found {boundary.numel()} — the prompt contract is broken"
                )
            labels[b, : int(boundary[0]) + 1] = IGNORE_INDEX
        labels[torch.isin(input_ids, register_id_tensor)] = IGNORE_INDEX
        if attention_mask is not None:
            labels[attention_mask == 0] = IGNORE_INDEX
        return labels

    def _head_batches(
        self, input_ids: torch.Tensor, utterances: list[Utterance], rendered
    ) -> dict[str, HeadBatch]:
        collected: dict[str, dict[str, list]] = {}
        # Batch-unique id per (batch row, word), so word-level heads can be paired up.
        word_ids: dict[tuple[int, int], int] = {}

        for b, (utterance, target) in enumerate(zip(utterances, rendered)):
            found: dict[str, list[int]] = {}
            for pos, token_id in enumerate(input_ids[b].tolist()):
                name = self.id_to_register.get(token_id)
                if name is not None:
                    found.setdefault(name, []).append(pos)
            cursor = {name: 0 for name in found}

            for slot in target.slots:
                positions = found.get(slot.name, ())
                i = cursor.get(slot.name, 0)
                if i >= len(positions):
                    raise RuntimeError(
                        f"utterance {utterance.index}: rendered target declares more "
                        f"{slot.name!r} registers than appear in input_ids "
                        f"({len(positions)}) — tokenization dropped a register"
                    )
                # Advance before the skip below, so an untrained field still consumes its
                # slot and the count check at the end of the loop stays meaningful.
                cursor[slot.name] = i + 1
                if (slot.level, slot.field) in UNTRAINED_FIELDS:
                    continue

                key = head_key(slot.level, slot.field)
                bucket = collected.setdefault(
                    key,
                    {"positions": [], "raw": [], "native": [], "phone_id": [], "word_index": []},
                )
                bucket["positions"].append((b, positions[i]))
                bucket["raw"].append(self._raw_value(utterance, slot))
                bucket["native"].append(self._native_value(utterance, slot))
                if slot.level == "phone":
                    bucket["phone_id"].append(self._phone_id(utterance, slot))
                elif slot.level == "word":
                    bucket["word_index"].append(
                        word_ids.setdefault((b, slot.word), len(word_ids))
                    )

            for name, positions in found.items():
                used = cursor.get(name, 0)
                if used and used != len(positions):
                    raise RuntimeError(
                        f"utterance {utterance.index}: {len(positions)} {name!r} registers "
                        f"in input_ids but {used} declared slots"
                    )

        batches: dict[str, HeadBatch] = {}
        for key, bucket in collected.items():
            level, field = key.split(".", 1)
            raw = torch.tensor(bucket["raw"], dtype=torch.float32)
            native = torch.tensor(bucket["native"], dtype=torch.float32)
            normalized = (native - self.stats.mean[key]) / self.stats.std[key]
            batches[key] = HeadBatch(
                positions=torch.tensor(bucket["positions"], dtype=torch.long),
                target=normalized,
                raw=raw,
                native=native,
                word_index=(
                    torch.tensor(bucket["word_index"], dtype=torch.long)
                    if level == "word"
                    else None
                ),
                phone_id=(
                    torch.tensor(bucket["phone_id"], dtype=torch.long)
                    if level == "phone"
                    else None
                ),
            )
        return batches
