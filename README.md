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

**Emitted scores and the loss mask**

- `TargetSpec.emit_scores` turns the register into a trigger. `render_target` then writes one grid token after every score register, so the score is in the text and later tokens attend to it. The renderer takes a grid-string to token map, because a score never reaches the text as digits: `1.8` is three tokens and sits off the grid.
- `prompting.slot_value` is the single definition of what a slot carries. The renderer reads it to pick the token, and the collator reads it to build the head target. Two copies would let the token and the head target disagree, and no test would notice.
- The mask treats a trigger and a score differently, and that is the whole point. A register is always masked: under the head path the model never emits one, and under a variant a trigger is punctuation fixed by the transcript. A score token is the prediction, so `Collator.score_ce` decides who supervises it.
- `score_ce=True` is `--ce-std` and `--ce-rail`. Cross-entropy reaches the token. `score_ce=False` is `--raft-rail` and `--veil-rail`. There the token is only a rendering of the expectation, so leaving it in the mask would re-add the mode-seeking term those arms exist to remove.
- A slot with no label emits **no token at all**. The synthetic negatives pair a transcript with unrelated audio, so their words have no stress pattern to judge. Inventing a score to keep the shape uniform would put a made-up number in the context of every later token. The register stays, the next register follows it, and nothing at that position is supervised.
- A variant builds no regression heads. The scores are in the text, so a head on the same field would be a second channel in one run. The arm would then not be the one documented under **Variants**. The collator still emits per-field positions and gold, because scoring and the RAFT loss both need them, so `HeadBank.read` skips keys the bank does not hold. With an empty bank the forward pass also stops requesting hidden states, which is the largest activation in the pass.
- Both arms emit byte-identical text. They differ only in the mask, so a difference in results cannot come from the data.

**Reading the grid and feeding it back**

- A score is not the token the model samples. It is the **expectation** of the next-token distribution, restricted to that type's grid and renormalised over it. `rail.GridSupport` does the restriction, the expectation and the snap, and derives every grid from `SCORE_GRID`. Mass on the rest of the vocabulary stays out of the denominator: leaving it in would shrink every prediction toward zero.
- The expectation is continuous and the stream is not, so it is **snapped** to the nearest grid token and written back. That reinjection is what makes this a unified model rather than a feature extractor with a regressor bolted on. `test_reinjection_changes_the_readout` asserts it: bias one phone score token and the last utterance score must move.
- `GridSupport.report` is what the metrics read, and it differs from the expectation for exactly one field. `word.stress` grids the vote count, so the readout is `P(votes >= 3)` mapped onto the published 5/10 median. The `binomial` head applies the same mapping, which is what keeps the two channels comparable.
- Scoring teacher-forces the transcript and generates only the scores. The published protocol fixes the phone sequence, so a model writing its own phones would misalign a few percent of registers and the phone correlation would stop meaning what the baselines mean. The reinjection stays real, because each emitted score changes what every later token attends to.
- The loop walks the turn left to right and feeds one segment per trigger, so the KV cache carries the rest. That is one forward pass per score rather than per token, and each pass adds the one to three tokens between two triggers.
- One utterance at a time. Reinjection makes the turn sequential, so a batch would have to advance in lockstep across utterances whose triggers sit in different places, and padding to the longest would write pad tokens into the cache and change what every later token attends to.
- `engine.score` dispatches on the collator, so no caller can pick the wrong channel. Scoring a RAIL run with the head reader would return an empty table rather than an error.

**The RAFT and VEIL terms**

- `--raft-rail` masks the score token out of the text loss and instead squares the error of the grid expectation. Cross-entropy rewards putting mass on the one token matching the label, which on a saturated field means collapsing onto the mode. A collapsed distribution has no expectation worth reading, and the expectation is the only thing RAIL reports.
- The grid term puts gradient on the neighbouring grid points a one-hot target leaves untouched. That is the arm's whole claim, and `--ce-rail` is the control that measures whether it is worth anything.
- Squared rather than log-cosh, unlike the head path. There the prediction is an unbounded scalar and one outlier residual dominates the batch. Here it is a mean over a fixed grid, so the residual cannot leave the grid's own span and its gradient is bounded already.
- The gold comes from `HeadBatch.raw`, and both sides are scaled by the **grid's** span rather than by `TargetStats`. `TargetStats` speaks the reported scale, which for `word.stress` is the 5/10 median while the grid holds the 0-5 vote count. Subtracting one from the other would compare two quantities and still return a tidy number.
- `--veil-rail` (**VEIL**, Variance & Expectation Informed Language loss) extends RAFT by applying the head path's anti-collapse composite objective directly to the restricted grid logits. Plain RAFT still faces variance collapse, endpoint softmax saturation at the grid maximum, and loss spikes on synthetic negatives (where `|r| = 2` squares to `4.0`, and `40.0` under 10× reweighting). VEIL replaces squared error with `log_cosh` on the normalized grid expectation and adds three terms:
  - **Label reweighting** (`LabelReweighter`, capped at `reweight_max = 10.0`) on `reweight_levels` (`phone` by default) and `word.stress`.
  - **Ceiling detection** (`lambda_detect * BCE(below ceiling)` with `detect_pos_weight = 4.3`) on `phone` and `word` grids, computed from `logsumexp(others) - z_top`. Utterance grids skip detection because 95% to 99% of utterance scores sit below `10`, matching the head path where utterance heads are `regression`.
  - **Balanced CCC** (`lambda_ccc * (1 - CCC)`, default `lambda_ccc = 0.05`) on the grid's `[-1, 1]` scale (`expectation_normalized` against `normalize_value(batch.raw)`). See **Balanced CCC** below.
- `GridSupport` takes logits **already restricted to the grid, in grid order**, everywhere. A function that guessed which of the two it was handed would be a bug waiting for the widths to be confused, and `(..., G)` is easy to pass where `(..., vocab)` was meant.
- The loss reads logits gathered at the **trigger**, because in a causal model the row at `p` is the distribution over `p + 1`, where the score token sits. Two tests close that gap from both sides: one pins the gather against the raw logits at the trigger, and one recomputes the model's own text loss with the shift and matches it. An off-by-one would leave every shape right and the loss still descending.
- The head path gathers nothing and pays nothing: `model.supports` is empty there, and `with_lm_loss=False` skips the gather entirely.

**Ordering**

- The model uses one unified turn: phones → word registers → utterance registers. It uses causal masking from the bottom up. Word registers see their phones, and utterance registers see all previous data.

**Reading**

- Each level uses a learned softmax mixture over the last 8 hidden states. This is biased low for phones and high for utterances. Lower layers predict fine-grained scores better.
- The mixture is specifically a weighted average of the last 8 transformer hidden states (via `LayerMixture`). The amount of weight is determined by learnable logits for each level (utterance, word, phone), with a softmax applied to determine how much each of the last 8 layers contributes.
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
- **Balanced CCC (`--heads-bccc` and `--veil-rail`)**: Detaching `N` buffer entries leaves only the `M` live elements carrying gradient, which scales the CCC gradient by `M / (N + M)`. At `batch_size = 1` and `N = 512`, that attenuates CCC by `~28×` on phones (`M ≈ 19`), `~103×` on words (`M ≈ 5`), and `513×` on utterances (`M = 1`), and breaks equivalence between `batch_size` and `grad_accum`. A `512`-slot utterance buffer also spans `128` optimizer steps at effective batch `4`, creating stale moments. Balanced CCC applies three fixes while leaving vanilla `heads` untouched:
  - **Straight-through `(N + M) / M` gradient compensation**: Multiplies the CCC backward gradient by `len(augmented) / len(live)` while keeping the logged `1 - CCC` scalar on `[0, 2]`. This restores `O(1)` CCC gradient pull at every level and makes `batch_size = 1, grad_accum = 4` match `batch_size = 4, grad_accum = 1`.
  - **Per-level buffer capacity**: Keeps `512` for `phone` (~7 steps) and `word` (~25 steps, preserving rare `word.stress` examples), and sets `utterance` to `64` (`--bccc-utterance-capacity`, `16` optimizer steps). This cuts utterance staleness `8×` and prevents `513×` overshoot against old history.
  - **Vote-scale alignment on `word.stress`**: On `heads-bccc`, the `binomial` stress head computes CCC on `2 * sigmoid(z) - 1` (the exact normalized Beta-Binomial expected vote count) against the `0..5` vote count (`batch.raw`) on `[-1, 1]`, matching `veil-rail` instead of chasing the `99:1` published median.
- **`lambda_ccc` coefficient choices**: Because `(N + M) / M` removes the silent `28×` to `513×` attenuation, keeping `lambda_ccc = 0.5` would overpower the pointwise loss. Each variant therefore sets its own default when `--lambda-ccc` is omitted:
  - `heads` (`0.5`): kept frozen for baseline compatibility with the uncompensated buffer.
  - `heads-bccc` (`0.1`): on unit-variance (`z`-scored) head targets, compensated `lambda_ccc = 0.1` puts CCC at roughly `10%` of the `log_cosh` gradient at every level without overpowering pointwise accuracy.
  - `veil-rail` (`0.05`): on VEIL's `[-1, 1]` grid span, target standard deviation is smaller ($\sigma_y \approx 0.25\text{ to }0.35$). Scale-invariant CCC gradients grow as $1/\sigma_y$ while `log_cosh` shrinks with $\sigma_y$, so `0.05` keeps CCC balanced against `log_cosh` instead of dominating it.
- **Detection term** (`--lambda-detect`, enabled by default): This is binary cross-entropy on "is this label below the top of its scale". It uses a distribution the head already has, so it adds no parameters. For `soft_class`, it is `log(1 − p_top) − log(p_top)`; for the stress head, it is the log-odds that the panel was not unanimous. This adds +0.038 to phone PCC. For word stress, the event it trains on differs from what the table reports.
- The model uses label-frequency reweighting for phones only, with a dynamic range cap of 10×. Uncapped inverse frequency spans 4176×, which creates a sawtooth loss. Word stress is always reweighted because 80.0% of its vote counts are a clean sweep.

**Masking**

- The model masks language-model cross-entropy at register positions and audio positions. Supervising the constant `<|audio|>` id collapses representations at speech positions, which prevents the model from hearing.

**Training**

- The model uses LoRA r=32 α=64 on all attention and MLP projections. The Gemma 4 audio path is encoder-free; it projects a raw 40 ms frame straight into the embedding space. LoRA on language layers acts as the acoustic adapter.
- The model uses a micro-batch of 1 and gradient accumulation of 4. Activation memory depends only on the micro-batch.
- Validation uses a fixed, speaker-disjoint, stratified split from the training set and reports correlation instead of loss. The model never touches the test split during training.

### Data splits and speaker stratification

The published `test` split (2,500 utterances, 125 speakers) stays untouched during training and serves as the final evaluation set.

The `train` and `val` splits come from the published `train` split (2,500 utterances, 125 speakers, 20 utterances per speaker) via `speaker_split` in `src/tutokana/data.py`:

- **Fixed split seed (`42`).** `--data-val-split-seed` defaults to `42` and stays separate from `--train-seed`. Multi-seed studies vary model initialization, oversampling mix, epoch shuffle, and dropout while sharing one fixed train/validation partition.
- **Speaker disjointness.** Every utterance from a given speaker stays together in either `train` or `val`. No speaker appears in both splits.
- **Age cohorts.** Speechocean762 divides cleanly into children (`age <= 15`, 58 speakers) and adults (`age > 15`, 67 speakers) in the training split. Validation slots apportion proportionally across both cohorts by largest remainder (6 children and 6 adults for `--data-val-speakers 12`).
- **Average total score bins.** Within each age cohort, `speaker_split` sorts speakers by their mean utterance `total` score (with tie-breaking on age and speaker ID). It then divides each sorted cohort into equal-sized quantile bins, one bin per allocated validation slot (6 child bins and 6 adult bins).
- **Sampling.** The split draws one speaker uniformly from each `(age cohort, score bin)` cell with the fixed seed `42`. That yields 12 speakers (240 utterances) across both age groups and the full proficiency range, avoiding range restriction in validation correlation.

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

A variant changes the output channel or core loss mechanics rather than a single ablation knob. Every variant shares the data path, the turn shape and the evaluation.

**All five non-default arms run.** `--heads-bccc`, `--ce-std`, `--ce-rail`, `--raft-rail` and `--veil-rail` all train and score end to end:

| flag | output channel | training loss | inference | default `lambda_ccc` |
| --- | --- | --- | --- | --- |
| *(default, `heads`)* | regression heads | composite head loss (uncompensated CCC) | teacher-forced heads | `0.5` |
| `--heads-bccc` | regression heads | composite head loss + Balanced CCC | teacher-forced heads | `0.1` |
| `--ce-std` | emitted grid tokens | plain cross-entropy on the emitted score token | mode-seeking (argmax) over the grid | `0.5` *(unused)* |
| `--ce-rail` | emitted grid tokens | plain cross-entropy on the emitted score token | RAIL: expected value over the grid | `0.5` *(unused)* |
| `--raft-rail` | emitted grid tokens | RAFT: squared error of the grid expectation | RAIL: expected value over the grid | `0.5` *(unused)* |
| `--veil-rail` | emitted grid tokens | VEIL (RAFT extension): reweighted `log_cosh` expectation error + phone/word ceiling detection + Balanced CCC | RAIL: expected value over the grid | `0.05` |

Each emitted-score arm needs its own score term switched on: `--ce-std` and `--ce-rail` need `--lambda-lm` above zero, and `--raft-rail` and `--veil-rail` need `--lambda-raft` above zero, because for each one that term is the only score supervision the arm has. The config refuses these combinations rather than running for hours on an objective with no score in it. `--variant heads_bccc`, `--variant ce_std`, `--variant ce_rail`, `--variant raft_rail` and `--variant veil_rail` are the same dest, and the arms are mutually exclusive. `variant` is one closed field rather than a boolean per arm, because a run that was multiple arms at once has no meaning for `--resume` to report. A run directory written before the field existed resumes as `heads`.

A flag for an arm that cannot yet train refuses before anything loads, and names what is missing. That guard is empty now and stays as the mechanism: a parsed flag nothing reads would train the head model, record the variant in its own `config.json`, and be indistinguishable from a real result.

The four emitted-score arms share one mechanism. A register stops being a place to read a head and becomes a **trigger**. The model emits the trigger itself, then emits one score token for it, restricted to that field's grid. Under `--ce-std` that token is the argmax on the grid (naive mode-seeking autoregression); under `--ce-rail`, `--raft-rail` and `--veil-rail` it is the nearest grid value to the current expectation. The text stream therefore carries the model's own belief, and later tokens attend to it.

That reinjection is the point. Without it the model is a feature extractor with a regressor bolted on, which is the predictive-head design this project rejected. The emitted-score arms form a ladder that establishes the baseline, separates the two halves of the RAFT paper's argument, and extends RAFT with the composite anti-collapse objective:

- `--heads-bccc` keeps the regression-head channel and replaces uncompensated buffered CCC with **Balanced CCC** (`(N + M) / M` straight-through gradient compensation, utterance buffer `64`, vote-scale stress CCC, and `lambda_ccc = 0.1`).
- `--ce-std` is the naive autoregressive MLM baseline: plain cross-entropy training with greedy argmax decoding restricted to each field's grid tokens (and majority mapping `votes >= 3 -> 10` else `5` on `word.stress`). It measures what a standard multimodal LM outputs before either RAIL or RAFT is applied.
- `--ce-rail` changes only the decision rule. RAIL fixes how a value is read out, but it cannot repair a distribution that cross-entropy already made mode-seeking. On word stress this arm is **predicted to lose**, and a loss here is a result rather than a bug.
- `--raft-rail` changes the objective, so the trained distribution matches the one the readout assumes. This is where the paper's argument applies.
- `--veil-rail` extends `--raft-rail` with **VEIL** (Variance & Expectation Informed Language loss). It keeps RAIL inference, replaces squared error with `log_cosh` on the normalized grid expectation, and adds the head path's anti-collapse terms (capped inverse-frequency reweighting, phone/word ceiling detection, and Balanced CCC with `lambda_ccc = 0.05` and utterance buffer `64`).

Three properties the arms were built around:

- **Read the score from the grid distribution, not from the emitted token.** One sampled token keeps about half the achievable correlation, and 16 averaged samples keep about 87%. Report the expectation at the trigger position, and let the token be a rendering.
- **`word.stress` scores on the vote-count grid and reports on the 5/10 median.** The released label is the median of five annotators, and 99% of it is one value. Training on it recreates the collapse the `binomial` head exists to avoid. Reporting still maps through `P(votes >= 3)`.
- **Reinjection makes the coupling between fields bidirectional.** `--stress-siblings` detaches its gradient so a 1%-of-words loss cannot damage two working columns. A text stream has no such detach, and nothing in the emitted path replaces it. That protection is gone in both arms, and a run's notes should say so.

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
  rail.py         grid readout: restrict, expect, snap, report
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
