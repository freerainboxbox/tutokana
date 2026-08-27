"""The model: Gemma 4 with LoRA, a register embedding delta, and the score heads.

`google/gemma-4-12B-it` is unusual in a way that simplifies this a lot: **its audio path is
encoder-free**. The only audio tensor in the checkpoint is
`model.embed_audio.embedding_projection.weight`, shape [3840, 640]. With 640 samples per
token at 16 kHz, a raw 40 ms waveform frame is projected straight into the language model's
embedding space — there is no mel spectrogram, no conformer tower, and the 48 decoder layers
*are* the acoustic model. So LoRA on the language layers already adapts the acoustic
pathway, and `train_audio_projection` opens up the single 2.46 M-parameter front-end matrix
as an ablation rather than as a necessity. (E2B/E4B do keep a conformer tower; conclusions
about the front end will not transfer between them.)

`RegisterDelta` exists because of how the registers are trained. They are never sampled —
their positions are structurally determined and force-fed — so only their *input* embedding
matters, not their logit. Making the tied 262144 x 3840 embedding matrix trainable to move a
dozen rows would cost a gigabyte of optimiser state to update 0.005% of it. A zero-init
(K, 3840) additive delta at register positions does the same job for ~50 K parameters, and
starts training exactly at the pretrained rows, which already have unit norm.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .heads import HeadBank

DEFAULT_MODEL_ID = "google/gemma-4-12B-it"
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        return torch.float16
    return torch.bfloat16


class RegisterDelta(nn.Module):
    """A trainable additive delta on the input embedding of register tokens only.

    Applied through a forward hook on the base model's embedding module rather than by
    passing `inputs_embeds`: the multimodal path needs `input_ids` to know where to scatter
    the projected audio frames, and handing it embeddings instead would bypass that.
    """

    def __init__(self, register_ids: dict[str, int], hidden_size: int):
        super().__init__()
        self.token_ids = sorted(register_ids.values())
        # `lookup` maps a vocabulary id to a row of `delta`, or -1 for every ordinary token.
        lookup = torch.full((max(self.token_ids) + 1,), -1, dtype=torch.long)
        for row, token_id in enumerate(self.token_ids):
            lookup[token_id] = row
        self.register_buffer("lookup", lookup, persistent=False)
        self.delta = nn.Parameter(torch.zeros(len(self.token_ids), hidden_size))

    def forward(self, input_ids: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        clamped = input_ids.clamp(max=self.lookup.numel() - 1)
        rows = self.lookup.to(input_ids.device)[clamped]
        is_register = rows >= 0
        if not bool(is_register.any()):
            return embeddings
        gathered = self.delta.to(embeddings.dtype)[rows.clamp_min(0)]
        return embeddings + gathered * is_register.unsqueeze(-1).to(embeddings.dtype)

    def attach(self, embedding_module: nn.Module):
        """Patch `embedding_module`'s output in place; returns the removable handle."""

        def hook(_module, args, output):
            return self(args[0], output)

        return embedding_module.register_forward_hook(hook)


@dataclass
class ModelConfig:
    model_id: str = DEFAULT_MODEL_ID
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    train_audio_projection: bool = False
    gradient_checkpointing: bool = True
    head_layers: int = 8
    head_proj_size: int = 512
    head_dropout: float = 0.1
    layer_mixture: bool = True
    attn_implementation: str = "sdpa"


class TutokanaModel(nn.Module):
    """Base model plus the parts that make register scoring work.

    `forward` returns `(head_outputs, lm_loss)`. Hidden states are requested for every
    layer because the heads consume a learned mixture over a window of them, not just the
    last one.
    """

    def __init__(
        self,
        base,
        head_bank: HeadBank,
        register_delta: RegisterDelta,
        config: ModelConfig,
    ):
        super().__init__()
        self.base = base
        self.heads = head_bank
        self.register_delta = register_delta
        self.config = config
        self._delta_handle = register_delta.attach(base.get_input_embeddings())

    def forward(self, batch: dict) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        input_ids = batch["input_ids"]
        model_kwargs = {
            key: value
            for key, value in batch.items()
            if key
            in (
                "attention_mask",
                "input_features",
                "input_features_mask",
                "mm_token_type_ids",
                "token_type_ids",
            )
            and value is not None
        }
        outputs = self.base(
            input_ids=input_ids,
            labels=batch.get("labels"),
            output_hidden_states=True,
            use_cache=False,
            **model_kwargs,
        )
        head_outputs = self.heads.read(outputs.hidden_states, batch.get("heads", {}))
        return head_outputs, getattr(outputs, "loss", None)

    def trainable_parameters(self) -> tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    # -- persistence -----------------------------------------------------------------
    # Only the trained parts are saved. The base weights stay in the Hub cache; a run
    # directory is a few hundred megabytes of adapter plus a few of heads, so keeping runs
    # around costs almost nothing.

    def save(self, run_dir: Path) -> None:
        from safetensors.torch import save_file

        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.base.save_pretrained(run_dir / "adapter")
        extra = {
            f"heads.{k}": v.detach().cpu().contiguous()
            for k, v in self.heads.state_dict().items()
        }
        extra.update(
            {
                f"register_delta.{k}": v.detach().cpu().contiguous()
                for k, v in self.register_delta.state_dict().items()
            }
        )
        save_file(extra, str(run_dir / "heads.safetensors"))

    def load_trained(self, run_dir: Path) -> None:
        from safetensors.torch import load_file

        blob = load_file(str(Path(run_dir) / "heads.safetensors"))
        head_state = {
            k[len("heads.") :]: v for k, v in blob.items() if k.startswith("heads.")
        }
        delta_state = {
            k[len("register_delta.") :]: v
            for k, v in blob.items()
            if k.startswith("register_delta.")
        }
        self.heads.load_state_dict(head_state, strict=False)
        self.register_delta.load_state_dict(delta_state, strict=False)


def load_processor(model_id: str):
    """Processor with the register tokens already made atomic."""
    from transformers import AutoProcessor

    from .tokens import register_tokens

    processor = AutoProcessor.from_pretrained(model_id)
    register_tokens(getattr(processor, "tokenizer", processor))
    return processor


def build_model(
    config: ModelConfig,
    head_specs: dict[str, dict],
    levels: tuple[str, ...],
    register_ids: dict[str, int],
    device: torch.device,
    adapter_dir: Path | None = None,
) -> TutokanaModel:
    """Load the base model, attach LoRA (or a saved adapter), heads and register delta."""
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    dtype = resolve_dtype(device)
    base = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        dtype=dtype,
        attn_implementation=config.attn_implementation,
    )

    if adapter_dir is not None:
        base = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    else:
        base = get_peft_model(
            base,
            LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=list(LORA_TARGET_MODULES),
            ),
        )

    if config.gradient_checkpointing:
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()

    # The single audio front-end matrix. Opt-in: published work on Phi-4-multimodal found
    # unfreezing the audio front end *lost* to LoRA-only on every aspect but one noisy cell,
    # so this is an experiment, not a default.
    if config.train_audio_projection:
        for name, parameter in base.named_parameters():
            if "embed_audio" in name:
                parameter.requires_grad_(True)

    base_config = getattr(base, "config", None)
    base_config = getattr(base_config, "text_config", base_config)
    hidden_size = base_config.hidden_size
    # `output_hidden_states` yields one entry per layer plus the embedding output, so a
    # 48-layer model exposes 49. Clamp rather than fail: a shallow model (a toy config in a
    # test, a smaller Gemma 4 for iteration) should still run, just with a shorter window.
    available = int(getattr(base_config, "num_hidden_layers", config.head_layers)) + 1
    head_layers = max(1, min(config.head_layers, available))
    if head_layers != config.head_layers:
        print(
            f"[model] head_layers {config.head_layers} exceeds the model depth; "
            f"using {head_layers}"
        )
    head_bank = HeadBank(
        hidden_size=hidden_size,
        specs=head_specs,
        levels=levels,
        n_layers=head_layers,
        proj_size=config.head_proj_size,
        dropout=config.head_dropout,
        layer_mixture=config.layer_mixture,
    )
    register_delta = RegisterDelta(register_ids, hidden_size)

    model = TutokanaModel(base, head_bank, register_delta, config)
    model.heads.to(device=device, dtype=torch.float32)
    model.register_delta.to(device=device, dtype=torch.float32)
    model.base.to(device)
    return model


def save_run_metadata(run_dir: Path, payload: dict) -> None:
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "config.json").write_text(json.dumps(payload, indent=2, default=str))
