# tutokana

Computer-aided pronunciation training on [speechocean762](https://huggingface.co/datasets/mispeech/speechocean762):
per-phone, per-word and per-utterance scoring from a fine-tuned Gemma 4 12B.

Scores are not generated as text. The assistant turn carries a phone transcript interleaved
with **score-register tokens**, and small regression heads read the hidden state at those
register positions. The model still emits a readable structured turn; only the numbers move
out of the text channel.

## Results

Pearson correlation on the full 2500-utterance test split, teacher-forced.

| model | phone | w.acc | w.stress | w.total | u.acc | u.fluency | u.prosodic | u.total |
|---|---|---|---|---|---|---|---|---|
| GOPT (2022) | 0.612 | 0.533 | 0.291 | 0.549 | 0.714 | 0.753 | 0.760 | 0.742 |
| 3MH (2023) | 0.693 | 0.682 | 0.361 | 0.694 | 0.782 | 0.843 | 0.836 | 0.811 |
| HMamba (2025) | 0.739 | 0.708 | **0.366** | 0.718 | **0.807** | **0.848** | **0.843** | **0.829** |
| Qwen2-Audio LoRA | 0.380 | 0.510 | 0.110 | 0.520 | 0.690 | 0.740 | 0.730 | 0.720 |
| **this repo** | **0.743** | **0.727** | 0.161 | **0.746** | 0.785 | 0.788 | 0.781 | 0.821 |

Sources: GOPT arXiv:2205.03432; 3MH via HMamba Table 1; HMamba aclanthology 2025.naacl-long.98;
Qwen2-Audio arXiv:2509.15701.

Phone accuracy is the headline: the best previously published *generative* model reached
0.380, and 0.743 is at parity with the GOP-feature specialists. Single seed — the 95% interval
on phone is [0.732, 0.754], so this is parity with HMamba, not a separated win.

## Design

**Output channel**
- Scores live in regression heads on register-token hidden states, not in generated text.
  Cross-entropy over digits carries no notion of distance and is mode-seeking, which collapses
  prediction variance on saturated labels.
- Eleven unused Gemma 4 vocabulary entries serve as registers: one per phone, three per word,
  four per utterance, plus separators. Registering them as atomic special tokens leaves
  `len(tokenizer)` at 262144, so no embedding resize and no id drift. `<unused8>` (id 14) is
  skipped as a known llama.cpp degenerate-loop trigger.
- Registers are force-fed at inference, never sampled, so one forward pass reads all eight
  fields and malformed output is impossible. Only their *input* embedding matters, so a
  zero-initialised `(11, 3840)` `RegisterDelta` is trained instead of the tied 262144×3840
  embedding matrix.

**Ordering**
- One unified turn: phones → word registers → utterance registers, bottom-up under causal
  masking, so word registers see their phones and utterance registers see everything.

**Reading**
- Each level takes a learned softmax mixture over the last 8 hidden states, biased low for
  phone and high for utterance. Fine-grained scores are better predicted in lower layers.
- A register is read at its own position; the −1 shift belongs to language-model
  cross-entropy only.

**Heads**
- `RMSNorm → Linear(512) → GELU → Dropout → Linear(n_out)`, fp32.
- Phone is `soft_class`: 11 logits over the label's real support (0.0–2.0 in 0.2 steps), read
  out as an expectation. The released phone label is the *mean* of the five experts' `{0,1,2}`
  rubric scores — every value is exactly k/5 — and not the rubric itself, which is what most
  secondary sources report; three-classing it discards ~20% of the signal. The word and
  utterance fields aggregate by median instead, which is what singles out word stress.
- Word stress is `binomial`; the remaining fields are `regression`.
- Phone conditioning is FiLM: a per-phone scale and shift from a 2-parameter embedding, 134
  parameters total. One head per symbol is not offered — rare phones would get too few
  examples and share no statistics.

**Word stress**
- The corpus scores every utterance with five experts and releases their *median*. A word is
  published as 10 ("stress correct, or monosyllabic") as soon as three of the five say so, so
  the label has exactly two values and 99.0% of them are 10. Trained directly, that is a 99:1
  binary problem and a regressor on it converges to a constant.
- The individual verdicts survive in the corpus's own `scores-detail.json`, which HuggingFace
  does not ship. Recovering them makes the target *how many of the five agreed*, 0–5, and
  20.0% of training words are then not a clean sweep against the 1.0% the median calls wrong.
  `data.load_stress_votes` caches that file beside the model weights; alignment is positional,
  so every word is checked against its released median rather than trusted to line up.
- The head still predicts one number: **p**, the rate at which a single expert would call the
  stress correct. Everything reported is derived from it, so the published comparison is
  untouched — the readout is P(median = 10) = P(at least three of five agree), mapped onto the
  corpus's 5–10 scale. A per-expert head would be the same model: the five are anonymous and
  unaligned across utterances, so nothing can condition on which expert is speaking.
- The panel is **Beta-Binomial(5, νp, ν(1−p))**, not Binomial(5, p). A binomial treats the
  head's p as the truth; it is an estimate, and the word-to-word spread it cannot yet resolve
  appears as overdispersion. The practical difference is the tail: a binomial calls a
  unanimous dissent from a confident head a one-in-a-billion event, and that event is 0.1% of
  words, so a single word would own the step. The beta-binomial keeps the gradient bounded
  near 1 there. ν is the dial between them — raise it and the Beta collapses to a point mass,
  recovering the plain binomial exactly (`--stress-concentration 100`).
- Detection comes free from the same distribution — "at least one expert flagged this" is a
  20.0% base rate, against the 1.0% the published label would offer.

*Fitting ν.* It is a property of the corpus, not something the model learns: measured once on
the training split and then held fixed. The difficulty is that p is never observed — only k,
how many of the five experts said "correct". But the *spread* of k gives it away. Even if
every word had the identical agreement rate p̄, the counts would still vary, purely because
the panel is five people rather than infinitely many, and that variance is a closed form.
More spread than that can only mean words genuinely differ in p:

```
Var(k)  =  n·p̄(1−p̄)      +      n(n−1)·Var(p)
           \___________/            \_________/
           five people disagreeing   words differing
           even at a fixed rate      from each other
```

Rearrange for Var(p), then match it to a Beta — whose variance is m(1−m)/(ν+1) — so two
measured facts fix the two unknowns:

```python
import numpy as np
from tutokana.data import STRESS_RATERS, load_split

k = np.array([w.stress_votes for u in load_split("train") for w in u.words], dtype=float)
n = STRESS_RATERS                                                    # 5

p_bar = k.mean() / n                                                 # 0.9471
var_p = (k.var() - n * p_bar * (1 - p_bar)) / (n * (n - 1))          # 0.005204
nu    = p_bar * (1 - p_bar) / var_p - 1                              # 8.633
```

Moment matching rather than maximum likelihood: it is closed-form and auditable in four
lines, which is what a hardcoded constant should be. Read ν as dispersion — the observed
counts spread **1.42×** wider than five independent experts would manage, so the panel is
worth about **3.5** independent experts, not 5. `experiments/audit_data.py` prints all of it
from the live dataset. One caveat: fitted on the marginal, ν absorbs both real expert
ambiguity and word-to-word variation a trained model could in principle explain, so 8.6 is
the conservative end — it errs toward a softer loss.

**Objective**
- log-cosh on z-scored targets, plus `λ_ccc · (1 − CCC)`, plus a detection term, plus
  language-model cross-entropy on text positions.
- Concordance correlation rather than Pearson: CCC penalises shrunken variance and shifted
  mean, which is the failure being guarded against. Pearson is invariant to both.
- The correlation term draws its moments from a 512-pair detached FIFO, not from the batch,
  so batch size can be chosen for memory.
- **Detection term** (`--lambda-detect`, on by default): binary cross-entropy on "is this
  label below the ceiling", read off a distribution the head already carries — for
  `soft_class` as `log(1 − p_top) − log(p_top)`, for the stress head as the probability that
  the panel was not unanimous. Zero new parameters. Worth +0.038 phone PCC.
- Label-frequency reweighting on phone only, with the dynamic range capped at 10×. Uncapped
  inverse frequency spans 4176× here, which makes the loss a sawtooth. Word stress is
  reweighted regardless of the setting: 80.0% of its vote counts are a clean sweep.

**Masking**
- Language-model cross-entropy is masked off register positions *and* audio positions.
  Supervising the constant `<|audio|>` id collapses the representations at exactly the
  positions carrying speech, producing a model that cannot hear.

**Training**
- LoRA r=32 α=64 on all attention and MLP projections. The Gemma 4 audio path is
  encoder-free — a raw 40 ms frame is projected straight into the embedding space — so LoRA
  on the language layers *is* the acoustic adapter.
- Micro-batch 1 × grad-accum 4. Activation memory is set by the micro-batch alone.
- Validation is speaker-disjoint from training and reports correlation, not loss. The test
  split is never touched during training.

### The assistant turn

```
<|turn>model
<|channel>thought
<channel|><task>
WE<sep_p>W<phn> IY0<phn><sep_w><w_acc><w_str><w_tot>
CALL<sep_p>K<phn> AO0<phn> L<phn><sep_w><w_acc><w_str><w_tot>
<u_acc><u_pro><u_flu><u_tot><turn|>
```

Verbatim apart from the register names: the only whitespace is what is shown, and `<sep_p>`
and `<sep_w>` are distinct tokens. Each word's registers follow that word's phones, behind
`<sep_w>`; the utterance registers follow every word, with the total last. Under causal
masking this reads bottom-up, so a word register attends to its own phones and the utterance
registers attend to all of them — the phone → word → utterance hierarchy HMamba's ablation
credits for its margin over a flat readout.

## Evaluation

Flattened Pearson per aspect per level over the whole split, matching the published protocol.
`--lite` reports only that. The full table adds four things a bare correlation hides:

- **95% bootstrap interval** on Pearson.
- **Spearman**, because Pearson is inflated by this corpus's skewed marginals.
- **σ_pred / σ_gold**, which exposes variance collapse a correlation can hide.
- **Detection** — see below.

### Spearman is a detection metric here

A continuous readout never ties; the labels are heavily tied. So even a *perfect* continuous
predictor is capped:

| field | modal share | SCC ceiling |
|---|---|---|
| phone.accuracy | 81.3% | 0.680 |
| word.accuracy | 89.7% | 0.528 |
| utterance.accuracy | 38.8% | 0.959 |

Word Spearman looks catastrophic beside utterance Spearman only because word labels are 90%
tied at 10. And decomposing what the ceiling is made of, on phone accuracy:

| | SCC |
|---|---|
| perfect predictor (the ceiling) | 0.680 |
| perfect ceiling-vs-not detection, random ordering below | 0.671 |
| perfect ordering below, ceiling not detected | 0.077 |

Detection is 99% of it. So the reported detection table (AUC, best-F1, base rate) is the
number to read, and a differentiable ranking loss would optimise orderings the metric cannot
reward. Fields with no dominant ceiling are omitted from that table — utterance scores span
2–10 with no modal value, so "below the maximum" is 95–99% of them and F1 is maximised by
calling everything positive.

### `--snap`

Optionally rescores predictions rounded onto the training label grid, which is persisted in
`target_stats.json`. Gold is quantised (0.2 for phone, integers elsewhere) and a continuous
readout is not; snapping restores those ties and removes error below half a grid step. It is
applied after the fact, is monotone, and cannot invent ordering the predictions lacked.

It is a per-field decision. Measured on the current run: word accuracy **+0.171 SCC**, word
total **+0.144**, both with PCC flat; phone and utterance slightly negative. Fields with a
two-value support are skipped — snapping onto `{5, 10}` is a threshold, not a rounding.

### Known caveats

- **Completeness is not trained.** It is 10.0 for 99.6% of the corpus; every published model
  reports NaN.
- **Word stress is near its ceiling, and is partly a re-measurement of word accuracy.** The
  metric largely reads annotation noise: resampling the five verdicts gives a gold label
  correlating 0.464 with the released one, and a model that knew every word's true
  expert-agreement rate would score about 0.50 against the published median — 0.42 treating
  agreement as one homogeneous population, 0.50 once syllable count is allowed to shift it, so
  read it as a floor. HMamba's 0.366 is ~74% of that.
- **Monosyllables are not a separable nuisance.** 84.9% of words have no lexical stress to
  misplace, yet the experts still split on 14.9% of them, and they split in a way that tracks
  general mispronunciation rather than stress: P(some expert dissents) is 11.2% when the word
  scores 10 for accuracy and 53.3% when it does not. On run-20260827-013850 this model's
  *word accuracy* prediction detects the published "stress wrong" label on monosyllables at
  AUC 0.984, where its own stress head manages 0.429. Those words carry 40.9% of the corpus's
  stress errors, so excluding them from the loss costs about 0.06 of the ceiling above; the
  column has to be trained on whole, it is simply not mostly measuring stress.
- **Single seed.** Differences below ~0.02 on this test set are seed noise.
- Reported single-rater inter-annotator agreement is ~0.555 at phone level, so published
  correlations above that are partly fitting the 5-rater averaging artifact.

## Usage

```bash
uv sync
python train.py                                  # current best configuration
python evaluate.py                               # full diagnostic table
python evaluate.py --lite                        # for presentation
python evaluate.py --snap --out runs/decompose.json
python train.py --resume <run-id>                # continue an interrupted run
```

Interrupting with Ctrl-C finishes the current step, checkpoints, and prints the resume
command. Resume restores optimizer moments, schedule position, RNG streams and the
correlation buffer; the mix, splits and epoch orders are recomputed from the saved config.
Eval is also interruptible and reports a partial table.

Base weights download to the standard Hugging Face cache, never the repo — set `HF_HOME` to
move it. `runs/` and `logs/` are gitignored and created on demand.

## Ablations

Each is a flag, not a rewrite. The defaults are the current best configuration.

| question | flag |
|---|---|
| is the detection term doing the work? | `--lambda-detect 0` |
| is the correlation term doing anything? | `--lambda-ccc 0` |
| does predicting every granularity at once cost the coarser ones? | `--data-levels utterance` |
| expectation over a discrete support, or plain regression? | `--head-modes-phone regression` |
| does reading lower layers help the fine levels? | `--no-layer-mixture` |
| is per-phone conditioning worth it? | `--phone-conditioning none` / `concat` |
| does reweighting the regression heads help or hurt? | `--reweight-levels phone,word` |
| is the overdispersed stress panel worth it, or would a binomial do? | `--stress-concentration 100` |
| does opening the audio front end help? | `--train-audio-projection` |

## Layout

```
src/tutokana/
  tokens.py       register table and registration
  data.py         dataset loading, targets, statistics
  mix.py          oversampling and synthetic negatives
  prompting.py    prompt and assistant-turn rendering
  collate.py      batching, register positions, loss masking
  heads.py        score heads, layer mixture, conditioning
  model.py        base + LoRA + register delta + heads
  losses.py       the composite objective
  engine.py       preflight, training, scoring, resume
  metrics.py      correlations, detection, ceilings
  reporting.py    result tables
  progress.py     console bar, file-only log detail
  config.py       frozen config tree, argparse binding, logging
train.py          training only
evaluate.py       evaluation only
experiments/      probes and one-off analyses
tests/
```
