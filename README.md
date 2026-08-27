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

## Architecture

### Base

`google/gemma-4-12B-it`, bf16, frozen except for LoRA. 48 decoder layers, hidden 3840, vocab
262144, tied embeddings.

Gemma 4 Unified is **encoder-free for audio**: the only audio tensor in the checkpoint is
`embed_audio.embedding_projection`, `[3840, 640]`. A raw 40 ms frame of 16 kHz waveform (640
samples) is projected straight into the embedding space — no mel spectrogram, no conformer
tower. The 48 decoder layers *are* the acoustic model, so LoRA on the language layers already
adapts the acoustic pathway. speechocean762 clips are short (p50 3.5 s, p95 8.1 s), giving
~89 audio tokens median against a 750-token budget, so full sequences run 400–700 tokens.

| trained | shape | notes |
|---|---|---|
| LoRA r=32, α=64, dropout 0.05 | `q,k,v,o,gate,up,down` × 48 layers | the acoustic adapter, given the above |
| register delta | (11, 3840) fp32, zero-init | see below |
| 8 score heads | 1.97 M each, fp32 | see below |
| audio projection | [3840, 640] | opt-in, `--train-audio-projection` |

### Register tokens

Eleven of Gemma 4's 6227 `<unusedN>` entries, registered as atomic special tokens. The rows
already exist, so `len(tokenizer)` stays 262144 — **no embedding resize, no id drift**.
`<unused8>` (id 14) is skipped: it is a known llama.cpp degenerate-loop trigger for Gemma 4.

| register | token | id | count per utterance |
|---|---|---|---|
| task | `<unused0>` | 6 | 1 |
| phone accuracy | `<unused1>` | 7 | one per phone (~19) |
| word accuracy / stress / total | `<unused2..4>` | 8–10 | three per word (~6) |
| utterance accuracy / prosodic / fluency / total | `<unused5..7>`, `<unused9>` | 11–13, 15 | 4 |
| phone / word separators | `<unused10..11>` | 16–17 | — |

Registers are **never sampled**. Their positions are structurally determined, so at inference
they are force-fed and one forward pass reads every head. Only their *input* embedding
matters, so instead of unfreezing the tied 262144 × 3840 matrix to move eleven rows,
`RegisterDelta` adds a trainable (11, 3840) tensor at register positions via a forward hook on
the embedding module — a hook rather than `inputs_embeds`, because the multimodal path needs
`input_ids` to know where to scatter the projected audio. Zero-init, so training starts
exactly at the pretrained rows (which already carry unit norm).

`utterance.completeness` has no register. It is 10.0 for 99.6% of train (sd 0.11); it is
measured and reported, never trained.

### Reading the hidden state

A register at index *i* is read **at its own position**. The state there already includes
token *i*; the −1 shift belongs to language-model cross-entropy, which predicts the next
token, and applying it here would read whatever precedes the register instead.

Each level pools its own learned convex combination over the last 8 hidden states
(`softmax` over a free parameter, initialised as `−2·|position − bias|` with bias 0.0 for
phone, 0.5 for word, 1.0 for utterance). HMamba's ablation is explicit that phone- and
word-level scores are better predicted from lower layers — a flat last-layer readout scores
0.694 phone correlation against 0.739 hierarchical — so the bias is an initialisation, not a
constraint, and the converged weights are a free diagnostic (printed at the end of training).

### Head body

Shared by all three modes, fp32:

```
RMSNorm(3840) → Linear(3840, 512) → GELU → Dropout(0.1) → Linear(512, n_out)
```

| level | mode | n_out | loss |
|---|---|---|---|
| utterance acc/pro/flu/tot | `regression` | 1 | log-cosh on the z-scored target |
| word accuracy, total | `regression` | 1 | log-cosh on the z-scored target |
| word stress | `binary` | 1 | BCE-with-logits, inverse-frequency weighted |
| phone accuracy | `soft_class` | 11 | cross-entropy, label smoothing 0.05 |

Targets are z-scored per field with train-split statistics, persisted to the run directory
and restored at evaluation. Correlation is affine-invariant so this is free at report time,
and it equalises gradient scale across fields whose native sd differ ~4× (utterance 1.5 on
0–10 against phone 0.36 on 0–2). Standard deviation is floored at 2% of the field's span,
which bounds the worst-case z-score near 50; anything past 60 fails preflight, because a
field that flat means the split is too small or the labels are not what they are assumed
to be.

### The phoneme regressor, specifically

This is the least obvious part, and the one that differs most from the obvious design.

**It is not a regressor.** 80.1% of phone accuracy labels are exactly 2.0. Under any
pointwise loss a scalar regressor drifts to that mode, which is precisely how the predecessor
reached 0.358 with a prediction spread 3% narrower than gold. So the head emits **11 logits
over the label's actual discrete support** — `{0.0, 0.2, …, 2.0}`, because the released label
is the mean of five annotators' `{0,1,2}` rubric scores, not the rubric itself. (Nearly every
secondary source repeats the `{0,1,2}` claim; three-class models quantise away about a fifth
of the label's real variance.)

Supervision is cross-entropy against the nearest support index with label smoothing 0.05.
Readout is the **expectation** `Σ p_k · v_k`, so the prediction is continuous and unconstrained
by the class grid — which is what a correlation metric needs — while the loss keeps the
calibrated, class-weightable structure that a scalar head cannot offer.

**Per-phone conditioning is FiLM, not one head per symbol.** The obvious reading of "a
regressor per phone" is 66 independent heads. That is the wrong shape: rare symbols get too
few examples, no statistics are shared across the trunk, and routing by the *gold* phone in
training but the *emitted* phone at inference introduces a train/test skew. Instead a single
`nn.Embedding(n_phones + 1, 2)` supplies a scale and shift applied to the head's output:

```
y = MLP(h) · (1 + γ_p) + β_p
```

Zero-initialised, so conditioning starts as an exact identity and only earns its keep if the
per-phone difficulty prior is real. Index 0 is a shared fallback for symbols absent from
training, so an unseen phone degrades instead of crashing. 134 parameters in total against
1.97 M for a single dedicated head, and every phone still trains the same trunk.

`--phone-conditioning` takes `none`, `film` (default) or `concat` — the last appends a
32-dimensional learned per-symbol embedding to the head's *input*, so the trunk itself can
behave differently per phone rather than only rescaling its output. A head per symbol is
deliberately not offered; anything else is rejected rather than silently degrading to `none`,
which would make an ablation arm quietly measure nothing.

### Label reweighting

Rare labels are upweighted by inverse frequency, normalised to mean 1 under the training
marginal so that enabling it does not implicitly change the effective learning rate.

**Which heads get it: phone only, by default.** Reweighting answers a classification
question — *this decision boundary sees too few examples of that class* — and only the phone
head is a classifier. On a `regression` head a label weight multiplies a log-cosh
**residual**, which rebalances nothing; it only declares that being wrong about a rare label
costs more, and on this corpus the rarest labels are also the noisiest (word accuracy 9
occurs 3 times, and each score is a 5-annotator mean). So the default is
`--reweight-levels phone`.

Two things sit outside that default:

- **`word.stress` is always reweighted**, whatever `--reweight-levels` says. It is a `binary`
  head at a 99:1 split — a detection problem, not a regression one — and there is no sane
  configuration in which an unbalanced stress head is the intended experiment.
- **`--reweight-levels phone,word`** restores the previous behaviour, as an ablation.

**How much: the ratio is capped.** Mean 1 constrains the average and leaves the tail free,
which is not enough — these tails are extreme. Raw inverse frequency spans a **4176×**
dynamic range on word accuracy, so one word in one micro-batch outweighs several hundred
ordinary ones. That is the difference between a loss curve and a sawtooth: the run that
exposed this oscillated between 10 and 148 around a median of 50, against a preflight loss
of 15.

| field | uncapped max weight | uncapped range | capped max (10×) |
|---|---|---|---|
| phone.accuracy | 92.0 | 810 | 3.58 |
| word.accuracy | 475.9 | 4176 | 4.76 |
| word.total | 132.2 | 1026 | 4.46 |
| word.stress | 50.3 | 100 | 9.18 |

`--reweight-max` (default 10) bounds the **ratio** between the largest and smallest weight,
and is applied *before* the mean-1 pass. Order matters: clamping afterwards does not hold,
because restoring mean 1 scales the clamped values straight back above the ceiling. Labels
rarer than the ceiling allows tie there, which is intended — beyond that point the
distinction is between noise and noise.

`--reweight-strength` is the exponent: 0 is uniform, 1 is full inverse frequency.
`--reweight-max 0` disables the cap, which is only useful for reproducing a run that predates
it.

### Objective

```
L = Σ_level  w_level · [ mean(pointwise) + λ_ccc · (1 − CCC) ]  +  λ_lm · CE(text)
```

with `λ_ccc = λ_lm = 0.5` and each level's loss normalised by element count before its weight
— otherwise phones (~19 per utterance) outvote the four utterance registers twenty to one.

The correlation term is **Lin's concordance**, not Pearson:

```
CCC = 2·cov(x,y) / (var(x) + var(y) + (mean(x) − mean(y))²)
```

Pearson is invariant to both variance shrinkage and mean shift, i.e. invariant to exactly the
failure being corrected. CCC penalises both.

A correlation needs a population and one batch is not one — utterance-level CCC over a batch
of 4 is noise. `CorrelationBuffer` keeps a detached FIFO of the last 512 (prediction, target)
pairs per head and computes the statistic over `live batch ∪ buffer`. Gradient flows only
through the live batch; the moments are estimated on hundreds of points. This is what lets
the batch size be chosen for memory rather than for the objective.

### Batching

`--train-batch-size` is the micro-batch; `--train-grad-accum` is how many of those are
accumulated per optimizer step. Effective batch is their product, 2 × 2 = 4 by default.
Activation memory is set by the micro-batch alone, so `grad_accum` buys a larger effective
batch for time rather than memory.

Language-model cross-entropy covers the transcript text only. It is masked off the registers
(nothing to learn there, and it would compete with the heads) and off the audio positions.
The second is not optional: the predecessor once trained a functionally deaf model — identical
output for real, zeroed, noised and swapped audio — because ~10% of its supervised tokens were
audio placeholders whose label is a constant, and that constant target collapsed the
representations at exactly the positions carrying the speech.

## Evaluation criteria

### Protocol

Pearson correlation per aspect per level, flattened over the whole test split — all 2500
utterances, ~16k words, ~47k phones, not macro-averaged per utterance. That is the protocol
GOPT established and every paper since has followed, so it is the number that goes in the
comparison column.

Scoring is **teacher-forced**: the canonical phone sequence comes from the dataset, so every
register aligns one-to-one with its gold score, and one forward pass reads all eight fields.
This is both the fast path (minutes, against the predecessor's ~30 h constrained decode) and
the comparable one. Letting the model generate its own phone sequence would leave a few
percent of positions misaligned, and the usual fix — skipping misaligned words — silently
flatters the phone correlation. Generation is measured separately, on a stratified subset,
as phone error rate (with and without stress digits) and word exact-match.

Test subsets are stratified, never a contiguous prefix: half evenly spaced over the split
sorted by utterance total, half over the mispronunciation-carrying subset. A prefix is
score-range-restricted and understates correlation.

### Reported columns, and why

| column | why |
|---|---|
| **PCC** | the field's standard metric; the only one that compares to published work |
| **95% CI** | percentile bootstrap. Differences below ~0.02 on this test set are inside it, and the literature routinely reports smaller single-seed gaps as improvements |
| **SCC** | Spearman. Both recent Microsoft papers argue Pearson is inflated on this corpus's skewed marginals — on their private set Pearson runs 0.87–0.95 while Spearman sits at 0.57–0.62 |
| **MSE / MAE** | scale-aware error, which correlation discards |
| **sd_pred / sd_gold** | the column that exposed the original problem. A model that has learned the marginal and nothing else can still post a respectable correlation while this sits near 0 |

PCC is reported as `nan` when either side is constant. That is the honest answer and is
exactly how the collapse surfaced before — it is not rounded to zero.

### Caveats carried deliberately

- **`utterance.completeness`** is 10.0 for 99.6% of train. Every LMM paper reports NaN; only
  GOP-feature models get 0.15–0.37, with standard deviations as large as the mean. It is
  measured and printed, never trained, and never tuned against.
- **`word.stress`** is 5 or 10 with 1.0% at 5. Published SOTA is 0.366 and a fine-tuned
  Qwen2-Audio managed −0.01. It is reported on the corpus's 5–10 scale (correlation is
  unchanged by the affine map from the binary scale it trains on), but it is not a target.
- **Inter-annotator agreement is itself low** — community measurements put single-rater
  agreement near 0.555 phone and 0.675 utterance accuracy. Published phone correlations of
  0.69–0.74 already exceed it, so the top of this benchmark is partly fitting the five-rater
  averaging artefact.
- **Run at least three seeds** and report mean ± sd before calling one configuration better
  than another.

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
runs/<run_id>/       adapter, heads, target statistics, config — gitignored, created on demand
tests/
```

Base weights live in the standard Hugging Face cache, never in the repo — set `HF_HOME` to
move it. `runs/` is a working directory and is gitignored in full: a single run carries a
LoRA adapter, a ~63 MB head bundle and ~1.2 GB of optimizer state. Both `runs/` and `logs/`
are created on demand, so a fresh clone needs no setup beyond `uv sync`.

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
| is per-phone conditioning worth it, and how rich? | `--phone-conditioning none` / `concat` |
| does reweighting the regression heads help or hurt? | `--reweight-levels phone,word` |
| does opening the audio front end help? | `--train-audio-projection` |

The first is the most interesting: published work on both Qwen2-Audio and GPT-4o found that
asking for all three granularities in one generated object costs the coarser levels several
points of correlation. Whether that survives the move to separately weighted heads is the
open question this repo is shaped to answer.
