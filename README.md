# tutokana

Computer-aided pronunciation training on
[speechocean762](https://huggingface.co/datasets/mispeech/speechocean762), built on
`google/gemma-4-12B-it`.

## Why this exists

Pronunciation scores are ordinal. Training a language model to *emit* them as text
supervises them with cross-entropy, which treats "8" versus "7" as exactly as wrong as "8"
versus "2" — no notion of distance, and mode-seeking. On a corpus where 80% of phone scores
are the maximum and 88% of word accuracies are 10, the predictable result is variance
collapse. In the predecessor project (`gemma-4-12b-so762`) every field predicted with a
smaller spread than the labels, and two fields came out literally constant, making their
correlation undefined:

| level | field | PCC | pred sd / gold sd |
|---|---|---|---|
| utterance | accuracy | 0.639 | 0.78 |
| utterance | completeness | **nan** | 0.00 |
| word | accuracy | 0.371 | 0.74 |
| word | stress | **nan** | 0.00 |
| phone | accuracy | 0.358 | 0.97 |

`tutokana` moves the numbers out of the text channel. The assistant turn still emits a
readable word and phoneme transcript, but every score position is a dedicated **register
token** — one of Gemma 4's 6227 unused vocabulary entries — whose final hidden state feeds a
small regression head. Registers are never sampled; they are force-fed, so one forward pass
reads every score at once.

The bar is [HMamba](https://aclanthology.org/2025.naacl-long.98.pdf) (phone 0.739, word
0.708/0.366/0.718, utterance 0.807/0.848/0.843/0.829), not the paper this line of work
started from — that one reports no word-level or phoneme-level scores at all.

## The assistant turn

```
<|turn>model<|channel>thought<channel|>
<task>
WE   <sep>W<phn> IY0<phn>          <sep><w_acc><w_str><w_tot>
CALL <sep>K<phn> AO0<phn> L<phn>   <sep><w_acc><w_str><w_tot>
<u_acc><u_pro><u_flu><u_tot><turn|>
```

Each word's registers follow that word's phones; the utterance registers follow every word,
with the total last. Under causal masking this reads bottom-up, so a word register attends to
its own phones and the utterance registers attend to all of them — the phone → word →
utterance hierarchy HMamba's ablation credits for its margin over a flat readout.

Cross-entropy is trained on the transcript text only, and masked off both the registers and
the audio positions. That second part is not optional: the predecessor once trained a
functionally deaf model by supervising audio placeholder positions whose label is a constant.

## Layout

```
train.py             training only
evaluate.py          evaluation only
src/tutokana/        the package
  tokens.py            the register table — single source of truth
  data.py              loading, targets, speaker-disjoint splits, normalisation
  mix.py               entropy-greedy oversampling + synthetic negatives
  prompting.py         the one renderer both training and evaluation call
  collate.py           batching, register positions, per-head targets, loss mask
  heads.py             layer mixtures, the three output modes, FiLM conditioning
  losses.py            log-cosh, buffered concordance correlation, reweighting
  model.py             Gemma 4 + LoRA + register delta + heads
  metrics.py           correlations, dispersion, phone error rate
  reporting.py         the results table, with published baselines alongside
  engine.py            preflight, training loop, teacher-forced scoring
  config.py            frozen dataclass config, argparse binding, run logging
experiments/         one-off probes and ablations
logs/                train-<run_id>-<timestamp>.log, eval-<run_id>-<timestamp>.log
runs/<run_id>/       adapter, heads, target statistics, config
tests/
```

Base weights live in the standard Hugging Face cache, never in the repo. Set `HF_HOME` to
move it.

## Usage

```bash
uv sync --extra dev
uv run pytest                              # 86 tests, no model download

uv run python experiments/audit_data.py    # confirm the label distributions first

uv run python train.py                     # full run, all three levels
uv run python train.py --train-probe 400   # short run for hyperparameter iteration
uv run python evaluate.py                  # scores the most recent completed run
uv run python evaluate.py --run-id run-20260826-2117 --generative
```

`evaluate.py` defaults to the most recently *completed* run, resolved by the timestamp in
`runs/<id>/config.json` rather than by mtime, so copying or opening a directory cannot change
which run is current.

## Ablations the design is built to support

Each is a flag, not a rewrite.

| question | flag |
|---|---|
| does predicting every granularity at once cost the coarser ones? | `--data-levels utterance` / `utterance,word` |
| is the correlation term doing anything? | `--lambda-ccc 0` |
| regression or an expectation over the discrete support? | `--head-modes-phone regression` |
| does reading lower layers help the fine levels? | `--no-layer-mixture` |
| is per-phone conditioning worth it? | `--phone-conditioning none` |
| does opening the audio front end help? | `--train-audio-projection` |

The first is the most interesting: published work on both Qwen2-Audio and GPT-4o found that
asking for all three granularities in one generated object costs the coarser levels several
points of correlation. Whether that survives the move to separately weighted heads is the
open question this repo is shaped to answer.

## Reporting

`evaluate.py` prints and logs correlation, rank correlation, error, and — the column that
exposed the original problem — `sd_pred / sd_gold`, with a bootstrap interval on the
correlation. Rank correlation is reported because Pearson is inflated on this corpus's skewed
marginals. Differences below roughly 0.02 on this test set are inside the interval; run
several seeds before calling one configuration better than another.
