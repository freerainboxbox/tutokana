# tutokana

This tool provides computer-aided pronunciation training using the [speechocean762](https://huggingface.co/datasets/mispeech/speechocean762) dataset. It uses a fine-tuned Gemma 4 12B model to score phones, words, and utterances.

The model does not generate scores as text. Instead, the assistant turn contains a phone transcript with **score-register tokens**. Small regression heads read the hidden state at these register positions. The model still produces a readable structured turn, but the numbers move out of the text channel.

The RAIL/RAFT experiments break that rule on purpose: a register becomes a **trigger**, and the token after it carries the score. See **Score grids** below.

## Results

The table shows Pearson correlation on the full 2500-utterance test split. The results use teacher-forcing.

| model            | phone     | w.acc     | w.stress  | w.total   | u.acc     | u.fluency | u.prosodic | u.total   |
| ---------------- | --------- | --------- | --------- | --------- | --------- | --------- | ---------- | --------- |
| GOPT (2022)      | 0.612     | 0.533     | 0.291     | 0.549     | 0.714     | 0.753     | 0.760      | 0.742     |
| 3MH (2023)       | 0.693     | 0.682     | 0.361     | 0.694     | 0.782     | 0.843     | 0.836      | 0.811     |
| HMamba (2025)    | 0.739     | 0.708     | **0.366** | 0.718     | **0.807** | **0.848** | **0.843**  | **0.829** |
| Qwen2-Audio LoRA | 0.380     | 0.510     | 0.110     | 0.520     | 0.690     | 0.740     | 0.730      | 0.720     |
| **this repo**    | **0.740** | **0.720** | 0.201     | **0.740** | 0.779     | 0.792     | 0.781      | 0.813     |

Sources: GOPT arXiv:2205.03432; 3MH via HMamba Table 1; HMamba aclanthology 2025.naacl-long.98; Qwen2-Audio arXiv:2509.15701.

Phone accuracy is the main result. The best previous generative model reached 0.380. This model reaches 0.740, which is similar to GOP-feature specialists. Using a single seed, the 95% interval for phone accuracy is [0.732, 0.754]. This result is similar to HMamba and not a separate win.

## Design

**Output channel**

- Regression heads on register-token hidden states hold the scores, not generated text. Cross-entropy over digits does not track distance and is mode-seeking. This collapses prediction variance on saturated labels.
- Eleven unused Gemma 4 vocabulary entries act as registers: one per phone, three per word, four per utterance, and separators. These are atomic special tokens. This keeps `len(tokenizer)` at 262144, so the embedding size does not change and IDs do not drift. The model skips `<unused8>` (id 14) because it triggers degenerate loops in llama.cpp.
- On the regression-head path, the model receives registers at inference but never samples them. One forward pass reads all eight fields, so malformed output is impossible. Only the input embedding matters. The model trains a zero-initialized `(11, 3840)` `RegisterDelta` instead of the tied 262144×3840 embedding matrix. The RAIL/RAFT path drops this property: the model emits the registers as triggers, so malformed output becomes possible and the parser has to handle it.

**Score grids (RAIL/RAFT)**

- Under RAIL/RAFT the register becomes a trigger, and the score moves into the text channel. The model emits the score in one step. The grid distribution is the next-token distribution, limited to that type's tokens.
- The code **derives the sets per score type**. Nothing is listed by hand. `SCORE_GRID` in `tokens.py` holds one `(spacing, low, high)` per scored type. `ScoreGrid` derives the values, the rendering and the token set from that. `GRID_STRINGS` is the union. Widen a grid, or add a type: one line changes, and the token set follows.
- **Spacing decides the rendering, and the score type decides the spacing.** A step of 1 renders integers. A step of 0.2 renders one decimal place. So `1` and `1.0` are different tokens. `word.stress` votes share the integer set, and `phone.accuracy` uses the decimal set. All integer types share one set, and all decimal types share the other. A token therefore means the same number wherever the model emits it.
- This rule is not cosmetic. The corpus writes `2` for a phone score of 2.0, and that value is 80.1% of phone labels. `ScoreGrid.render` reads the score type, so it still emits the decimal token `2.0`. A formatter that read the float would move phone scores onto the integer tokens. That breaks the grid silently.
- `low` and `high` mark the ends of the **target space**, not the observed support. The training split omits word accuracy 4 and utterance accuracy 0 and 2, but those values exist. A regression-aware loss needs a grid that spans the space, so the expectation can land anywhere in it. Snapping is lossless on gold, because every released label sits on its own grid.
- Grid tokens use `special=False`, the same policy as the registers. See the two bullets under **Output channel** for why the split policy was dropped.
- Decoding uses `skip_special_tokens=False`. `parse_generated` walks the anchors instead of a hard-coded register list, so markers, triggers and score tokens all follow one rule. A default decode loses the field boundaries, so the parser cannot tell which field a score belongs to.
- A score decodes to its slot name, `<unused20>`, not to `8`. The channel is atomic and stable, but not human-readable, so the parser maps id to value through the grid table. `strip_anchors` exists for display only.
- `utterance.completeness` is deliberately gridless. It is untrained and has no register. Its released values (1.7, 6.7, 7.1, 7.5, 8.6, 8.9) sit on no step of 1 or 0.2. Training it would force a 0.1 grid and about 100 more tokens.
- Slots come from the low `unused` block, ids 18 to 39, and use 22 of 77 free slots. `RegisterDelta` sizes its lookup table by the largest id it sees. Slots at 256001+ would grow that table from tens of entries to a quarter of a million. `<unused8>` (id 14) stays excluded.
- Gemma 4 ties `lm_head.weight` to the embedding matrix, so a delta on the input activations alone never reaches the output side. `ModelConfig.emitted_delta` adds a second hook that puts `h . delta` into the logits of our columns. That is what makes the score tokens learnable in both directions. It is off by default, so the regression-head path is unchanged.
- A score token occupies an unused slot whose base row carries no numeric meaning. So the delta seeds from the digits the token stands for, and the row for `1.0` becomes the mean of `1`, `.` and `0`. The seed subtracts the slot's own base row, because the delta is added to it. Registers keep a zero init. A field name has no textual meaning to inherit.
- Delta rows are checkpointed **by vocabulary id**, alongside a persisted id list. `load_state_dict` raises on a size mismatch even with `strict=False`, and the row count goes from 11 to 33 the moment the grid tokens join. So `merge_state` copies rows by id instead. Rows the file does not cover keep their seed. That is what lets a variant warm-start from a regression-head checkpoint.

**Ordering**

- The model uses one unified turn: phones → word registers → utterance registers. It uses causal masking from the bottom up. Word registers see their phones, and utterance registers see all previous data.

**Reading**

- Each level uses a learned softmax mixture over the last 8 hidden states. This is biased low for phones and high for utterances. Lower layers predict fine-grained scores better.
- The model reads a register at its own position. The −1 shift is only for language-model cross-entropy.

**Heads**

- The architecture is `RMSNorm → Linear(512) → GELU → Dropout → Linear(n_out)` in fp32.
- Phone accuracy uses `soft_class`: 11 logits over the label's real support (0.0–2.0 in 0.2 steps). The model reads this as an expectation. The released phone label is the mean of five experts' `{0,1,2}` rubric scores. Every value is exactly k/5. This differs from the rubric itself reported in secondary sources; using three classes would lose about 20% of the signal. Word and utterance fields use the median instead, which highlights word stress.
- Word stress uses `binomial` logic; other fields use `regression`.
- Phone conditioning uses FiLM: a per-phone scale and shift from a 2-parameter embedding (134 parameters total). The model does not use one head per symbol. Rare phones would have too few examples and share no statistics.

**Word stress**

- Five experts score every utterance, and the corpus releases their median. A word is marked 10 ("stress correct, or monosyllabic") if three of five experts agree. This creates a binary label where 99.0% are 10. A regressor trained on this converges to a constant.
- Individual verdicts exist in `scores-detail.json`, but HuggingFace does not provide this file. Recovering these verdicts changes the target to "how many of the five agreed" (0–5). Then, 20.0% of training words are not a clean sweep, compared to the 1.0% that the median calls wrong. `data.load_stress_votes` caches this file with the model weights. The system checks every word against its released median to ensure alignment.
- The head predicts one number: **p**, the rate at which a single expert would call the stress correct. All reported values derive from this. The published comparison remains unchanged: P(median = 10) = P(at least three of five agree), mapped to the corpus's 5–10 scale. A per-expert head would be the same model because the five experts are anonymous and unaligned across utterances.
- The panel follows a **Beta-Binomial(5, νp, ν(1−p))** distribution, not a Binomial(5, p). A binomial treats the head's p as truth. However, p is an estimate, and word-to-word spread appears as overdispersion. The beta-binomial keeps the gradient bounded near 1 for unanimous dissent. ν controls the difference between these models; a high ν recovers the plain binomial (`--stress-concentration 100`).
- Detection is a free result of this distribution. "At least one expert flagged this" has a 20.0% base rate, while the published label only has 1.0%.
- Synthetic negatives have **no stress label**. Before 2026-09-01, these were marked as a clean sweep. This put about 1,800 words per epoch at the low end of the mix (word total ≤ 1, every phone at 0) but labeled them "stress perfect". In contrast, only 453 real low-score words per epoch were labeled wrong. Real annotators call the stress wrong on 32–41% of low-score words. The head learned this: on the test split, 40 words with total ≤ 1 are stress errors 100% of the time. The stress head ranked these above perfect words. This accounts for most of the gap between the stress head (0.128) and the word-total head (0.333). The model still emits their register to keep the turn shape, but the collator ignores them (`Word.stress_rated`).

*Fitting ν.* This is a property of the corpus, not a learned parameter. It is measured once on the training split and held fixed. The difficulty is that p is never observed; only k (the number of experts who said "correct") is known. The spread of k reveals p. Even if every word had the same agreement rate p̄, counts would still vary because the panel has five people. Variance beyond that means words differ in p:

```
Var(k)  =  n·p̄(1−p̄)      +      n(n−1)·Var(p)
           \___________/            \_________/
           five people disagreeing   words differing
           even at a fixed rate      from each other
```

Rearrange for Var(p), then match it to a Beta distribution. Two measured facts fix the two unknowns:

```python
import numpy as np
from tutokana.data import STRESS_RATERS, load_split

k = np.array([w.stress_votes for u in load_split("train") for w in u.words], dtype=float)
n = STRESS_RATERS                                                    # 5

p_bar = k.mean() / n                                                 # 0.9471
var_p = (k.var() - n * p_bar * (1 - p_bar)) / (n * (n - 1))          # 0.005204
nu    = p_bar * (1 - p_bar) / var_p - 1                              # 8.633
```

This uses moment matching instead of maximum likelihood because it is closed-form and auditable. ν represents dispersion. Observed counts spread **1.42×** wider than five independent experts would. Thus, the panel is worth about **3.5** independent experts, not 5. `experiments/audit_data.py` prints these values from the live dataset. One caveat: ν absorbs both real expert ambiguity and word-to-word variation. The value 8.6 is conservative and results in a softer loss.

**Objective**

- The objective uses log-cosh on z-scored targets, plus `λ_ccc · (1 − CCC)`, a detection term, and language-model cross-entropy on text positions.
- The model uses Concordance correlation (CCC) instead of Pearson. CCC penalizes shrunken variance and shifted means, which prevents specific failures. Pearson is invariant to both.
- The correlation term uses moments from a 512-pair detached FIFO, not the batch. This allows you to choose the batch size based on memory.
- **Detection term** (`--lambda-detect`, enabled by default): This is binary cross-entropy on "is this label below the top of its scale". It uses a distribution the head already has, so it adds no parameters. For `soft_class`, it is `log(1 − p_top) − log(p_top)`; for the stress head, it is the log-odds that the panel was not unanimous. This adds +0.038 to phone PCC. For word stress, the event it trains on differs from what the table reports.
- The model uses label-frequency reweighting for phones only, with a dynamic range cap of 10×. Uncapped inverse frequency spans 4176×, which creates a sawtooth loss. Word stress is always reweighted because 80.0% of its vote counts are a clean sweep.

**Masking**

- The model masks language-model cross-entropy at register positions and audio positions. Supervising the constant ` također` id collapses representations at speech positions, which prevents the model from hearing.

**Training**

- The model uses LoRA r=32 α=64 on all attention and MLP projections. The Gemma 4 audio path is encoder-free; it projects a raw 40 ms frame straight into the embedding space. LoRA on language layers acts as the acoustic adapter.
- The model uses a micro-batch of 1 and gradient accumulation of 4. Activation memory depends only on the micro-batch.
- Validation uses speakers that are disjoint from training and reports correlation instead of loss. The model never touches the test split during training.

### The assistant turn

```
WE<sep_p>W<phn> IY0<phn><sep_w><w_acc><w_str><w_tot>
CALL<sep_p>K<phn> AO0<phn> L<phn><sep_w><w_acc><w_str><w_tot>
<u_acc><u_pro><u_flu><u_tot>
```

The turn is verbatim except for the register names. The only whitespace is what is shown. `<sep_p>` and `<sep_w>` are distinct tokens. Word registers follow the word's phones after `<sep_w>`; utterance registers follow all words, with the total last. Causal masking reads this bottom-up: a word register sees its phones, and utterance registers see all previous data. This phone → word → utterance hierarchy is why this model outperforms a flat readout, as seen in HMamba's ablation.

## Evaluation

The model calculates flattened Pearson per aspect per level over the whole split, following the published protocol. `--lite` reports only that. The full table adds four metrics:

- **95% bootstrap interval** on Pearson.
- **Spearman**, because Pearson is inflated by skewed marginals in this corpus.
- **σ_pred / σ_gold**, which shows variance collapse that correlation might hide.
- **Detection** — see below.

### What "detection" means

One binary event is fixed: **is this label below the top of its scale?** This means "is this not a perfect score". The ceiling is the field's maximum. A label either sits on it or it does not.

| field          | ceiling | the event                                   | base rate |
| -------------- | ------- | ------------------------------------------- | --------- |
| phone.accuracy | 2.0     | this phone was not scored perfect           | 18.7%     |
| word.accuracy  | 10      | this word was not scored perfect            | 10.3%     |
| word.total     | 10      | this word was not scored perfect overall    | 11.9%     |
| word.stress    | 10      | the experts' median called the stress wrong | 0.8%      |

The `thresh` column in the eval table is different: it is where the model's continuous prediction is cut to create a yes/no call. This threshold is chosen after the fact to maximize F1. It changes F1, precision, and recall, but not AUC. Read the AUC column for a threshold-free metric.

**Word stress trains on a different event than it reports.** The loss uses the vote count, so its ceiling is 5 of 5. The event is "at least one of the five experts dissented", which has a 20.0% base rate. The table uses the published median, so its ceiling is 10 and the event is "the median called it wrong", which has a 0.8% base rate. The auxiliary term fits a 20% question, while the report scores a 0.8% one. This may not affect results, but it is not currently separable because `--lambda-detect` is global and phone accuracy requires the term.

### Spearman as a detection metric

A continuous readout never ties, but labels are heavily tied. A perfect continuous predictor is therefore capped:

| field              | modal share | SCC ceiling |
| ------------------ | ----------- | ----------- |
| phone.accuracy     | 81.3%       | 0.680       |
| word.accuracy      | 89.7%       | 0.528       |
| utterance.accuracy | 38.8%       | 0.959       |

Word Spearman is low compared to utterance Spearman because 90% of word labels are tied at 10. Decomposing the ceiling for phone accuracy shows:

|                                                         | SCC   |
| ------------------------------------------------------- | ----- |
| perfect predictor (the ceiling)                         | 0.680 |
| perfect ceiling-vs-not detection, random ordering below | 0.671 |
| perfect ordering below, ceiling not detected            | 0.077 |

Detection provides 99% of the score. Therefore, read the detection table (AUC, best-F1, base rate). A differentiable ranking loss would optimize orderings that this metric cannot reward. Fields without a dominant ceiling are omitted from the table because "below the maximum" is too common (e.g., 95–99% for utterance scores), making F1 maximize by calling everything positive.

### `--snap`

This optionally rescores predictions by rounding them onto the training label grid, which is in `target_stats.json`. Gold labels are quantized (0.2 for phone, integers elsewhere). Continuous readouts are not; snapping restores ties and removes error below half a grid step. This is applied after the fact, is monotone, and does not change the ordering of predictions.

Snapping is a per-field decision. In current runs: word accuracy **+0.171 SCC**, word total **+0.144**, with PCC remaining flat; phone and utterance results are slightly negative. Fields with a two-value support are skipped because snapping onto `{5, 10}` is a threshold, not rounding.

### Known caveats

- **Completeness is not trained.** It is 10.0 for 99.6% of the corpus; every published model reports NaN.
- **Word stress is near its ceiling and partly measures word accuracy.** The metric mostly reads annotation noise. Resampling the five verdicts gives a gold label that correlates 0.464 with the released one. A model that knew every word's true expert-agreement rate would score about 0.50 against the published median. This is 0.42 if agreement is one homogeneous population, and 0.50 when syllable count is included. HMamba's 0.366 is about 74% of that.
- **Monosyllables are not a separate problem.** 84.9% of words have no lexical stress, yet experts still split on 14.9% of them. This split tracks general mispronunciation: P(some expert dissents) is 11.2% when the word scores 10 for accuracy and 53.3% when it does not. On run-20260827-013850, the word accuracy prediction detects "stress wrong" labels on monosyllables at AUC 0.984, while the stress head manages 0.429. These words make up 40.9% of stress errors. The model still emits their register to keep the turn shape, but the collator ignores them (`Word.stress_rated`).
- **Single seed.** Differences below ~0.02 on this test set are due to the seed.
- Single-rater inter-annotator agreement is ~0.555 at the phone level. Published correlations above that may partly fit the 5-rater averaging artifact.

## Usage

```bash
uv sync
python train.py                                  # current best configuration
python evaluate.py                               # full diagnostic table
python evaluate.py --lite                        # for presentation
python evaluate.py --snap --out runs/decompose.json
python train.py --resume <run-id>                # continue an interrupted run
```

Press Ctrl-C to interrupt. The model finishes the current step, checkpoints, and prints the resume command. Resume restores optimizer moments, schedule position, RNG streams, and the correlation buffer. The system recomputes the mix, splits, and epoch orders from the saved config. Evaluation is also interruptible and reports a partial table.

Base weights download to the standard Hugging Face cache. Set `HF_HOME` to change this location. The `runs/` and `logs/` directories are gitignored and created automatically.

## Ablations & Variants

**Ablations**

Each ablation is a flag. The defaults use the current best configuration. Every one of these keeps the regression-head output channel, so they are all measured against the table in **Results**.

| question                                                            | flag                                   |
| ------------------------------------------------------------------- | -------------------------------------- |
| is the detection term doing the work?                               | `--lambda-detect 0`                    |
| is the correlation term doing anything?                             | `--lambda-ccc 0`                       |
| does predicting every granularity at once cost the coarser ones?    | `--data-levels utterance`              |
| expectation over a discrete support, or plain regression?           | `--head-modes-phone regression`        |
| does reading lower layers help the fine levels?                     | `--no-layer-mixture`                   |
| is per-phone conditioning worth it?                                 | `--phone-conditioning none` / `concat` |
| does reweighting the regression heads help or hurt?                 | `--reweight-levels phone,word`         |
| is the overdispersed stress panel worth it, or would a binomial do? | `--stress-concentration 100`           |
| is word stress just a re-measurement of word quality?               | `--stress-siblings`                    |
| does opening the audio front end help?                              | `--train-audio-projection`             |

**Variants**

A variant changes the output channel rather than a term inside it, so it is not a subtraction from the current model. It is a different model that shares the data path, the turn shape and the evaluation.

**Neither arm can run yet.** Both flags parse, are recorded in `config.json` and are diffed on resume, so a run states which channel it trained. What is not built is the training and inference path, so both arms stop before anything is loaded:

| flag | training loss | inference | status |
| --- | --- | --- | --- |
| `--ce-rail` | plain cross-entropy on the emitted score token | RAIL: expected value over the grid | parses, refuses |
| `--raft-rail` | RAFT: squared error of the grid expectation | RAIL: expected value over the grid | parses, refuses |

`--variant ce_rail` and `--variant raft_rail` are the same dest, and the arms are mutually exclusive. `variant` is one closed field rather than a boolean per arm, because a run that was both arms at once has no meaning for `--resume` to report. A run directory written before the field existed resumes as `heads`.

The refusal names what is still missing, which is the point of accepting the flag early: a parsed flag nothing reads would train the head model, write `variant: ce_rail` into its own record, and be indistinguishable from a real result.

Both arms share one mechanism. A register stops being a place to read a head and becomes a **trigger**. The model emits the trigger itself, then emits one score token for it. That token is the nearest grid value to the current expectation. The text stream therefore carries the model's own belief, and later tokens attend to it.

That reinjection is the point. Without it the model is a feature extractor with a regressor bolted on, which is the predictive-head design this project rejected. The two arms also separate the two halves of the RAFT paper's argument, and reading them needs both:

- `--ce-rail` changes only the decision rule. RAIL fixes how a value is read out, but it cannot repair a distribution that cross-entropy already made mode-seeking. On word stress this arm is **predicted to lose**, and a loss here is a result rather than a bug.
- `--raft-rail` changes the objective, so the trained distribution matches the one the readout assumes. This is where the paper's argument applies.

Three consequences to carry into any implementation:

- **Read the score from the grid distribution, not from the emitted token.** One sampled token keeps about half the achievable correlation, and 16 averaged samples keep about 87%. Report the expectation at the trigger position, and let the token be a rendering.
- **`word.stress` scores on the vote-count grid and reports on the 5/10 median.** The released label is the median of five annotators, and 99% of it is one value. Training on it recreates the collapse the `binomial` head exists to avoid. Reporting still maps through `P(votes >= 3)`.
- **Reinjection makes the coupling between fields bidirectional.** `--stress-siblings` detaches its gradient so a 1%-of-words loss cannot damage two working columns. A text stream has no such detach. That protection is gone in both variants, and the run notes should say so.

Comparing a variant against the **Results** table is not apples to apples. The table is teacher-forced. A variant feeds its own emitted tokens back, so it should be reported with its own row and its own protocol note.

## Layout

```
src/tutokana/
  tokens.py       register table, score grids, registration
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
