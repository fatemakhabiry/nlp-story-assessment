# Task 2: Genre Classification with Multi-Task Preservation — Documentation

## Executive Summary

Two full architectures were built, trained, and evaluated under identical
5-fold cross-validation splits for a direct, apples-to-apples comparison:

| | ModernBERT-base (separate model) | Qwen2.5-7B + QLoRA (shared instance) |
|---|---|---|
| Accuracy | 52.9% ± 3.2% | **63.8% ± 0.8%** |
| Macro-F1 | 50.0% ± 3.5% | **60.4% ± 1.2%** |
| Params trained | ~150M (full fine-tune) | tens of millions (LoRA + head) |
| Train time / fold | ~50 min | ~107 min |
| Forgetting risk | None (separate model) | Structurally proven near-zero (Section 8) |

**Qwen2.5-7B + QLoRA shows a statistically significant, large-effect-size
accuracy advantage** (paired t-test p=0.004, Cohen's d=2.65 — see Section 6.3)
and satisfies the brief's stated preference for a single shared model.
**Given this evidence, Qwen QLoRA is the stronger Task 2 classifier.**

---

## 1. Approach Summary

The dataset provides 1,000 stories across 99 unique genres. The source
`story_genres.pkl` file contains 100 entries but only 99 unique genre names
because **`Historical Adventure` appears twice**. All 99 unique genres in
the metadata occur in the story dataset; no genre was removed or invented.
This works out to roughly **10
labeled examples per class**, which shaped the initial hypothesis behind
this document's original architecture choice: that a smaller, purpose-built
discriminative encoder would be more sample-efficient than adapting a much
larger generative model's representations. **The evidence did not support
this hypothesis** — Qwen2.5-7B's richer pretrained representations appear
to outweigh the sample-efficiency advantage of a smaller encoder at this
data scale, based on the results in Section 6. Updating this conclusion in
light of the evidence, rather than anchoring to the initial hypothesis, is
itself part of the documented reasoning process here.

Two architectures were built and compared empirically:

1. **ModernBERT-base**, fully fine-tuned as a dedicated, purpose-built
   classification encoder — a separate model from Task 1 (Sections 4-7).
2. **Qwen2.5-7B via QLoRA, sharing the same loaded instance as Task 1's
   chatbot** (LoRA injected in place via `get_peft_model`, plus a custom
   classification head, trained via HuggingFace `Trainer`) — satisfying the
   brief's stated preference for a single multi-purpose model (Sections
   4B-8).

---

## 2. Data Preprocessing

**Genre label validation.** The source `story_genres.pkl` file contains 100
entries but only **99 unique genre names** because `Historical Adventure`
appears twice. Cross-checking the metadata against the dataset showed that
all 99 unique genre names occur in the 1,000 stories. Therefore, both models
use a 99-way classifier without deleting, merging, or inventing a class.

**Label encoding.** Genre strings encoded as integer class IDs via
`sklearn.LabelEncoder`, producing **99 classes**.

**Class balance.** Counts range from 10 to 20 examples per genre (mean
10.1); one outlier genre has 20 due to the duplicate mapping in the source
data. The remaining 98 genres are uniform at 10 each.

**Split strategy: stratified 5-fold cross-validation.** With only ~10
examples per genre, a single conventional train/val/test split would leave
1-2 examples per genre in validation — high-variance, unreliable metrics.
Both architectures use the **same fold splits**
(`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`), enabling a
valid paired statistical comparison (Section 6.3).

**Sequence length (ModernBERT):** measured empirically rather than assumed —
see coverage table in Section 4. **`MAX_LENGTH = 2048`** selected (95.3%
coverage).

**Sequence length (Qwen):** same `MAX_LENGTH = 2048`, tokenized with the
Qwen tokenizer, for consistency with Task 1's context handling.

---

## 3. Simple Baselines (for context, before model comparison)

| Approach | Training time | Accuracy | Macro-F1 |
|---|---|---|---|
| TF-IDF (1-2 grams) + Linear SVM | seconds | 44.5% | 41.3% |
| Frozen embeddings (BGE-small) + Logistic Regression | seconds | 45.0% | 39.4% |

Both land well above random guess (~1% for 99 classes) at near-zero cost —
a calibration point for how much a fine-tuned model needs to clear to
justify its cost.

---

## 4. ModernBERT Training Methodology

### 4.1 Model selection: why ModernBERT-base

| Model | Reason not selected |
|---|---|
| BERT-base / RoBERTa-base | Native max context is 512 tokens. Per the coverage analysis (Section 4.4), only 10.3% of stories fit fully within 512 tokens — these models would require aggressive truncation or a head+tail compromise, losing substantial content for most of the dataset. |
| DistilBERT | Faster and smaller, but generally has a lower classification-accuracy ceiling than full BERT-base-class models. Given classification quality — not inference speed — was the primary axis for Task 2 model selection (speed is Task 3's concern), this trade-off wasn't justified. |
| DeBERTa-v3-base | Also considered and directly attempted as a secondary comparison point (Section 4.6). Shares BERT/RoBERTa's 512-token native context limitation, and the attempted run additionally hit a training-stability issue. |
| **ModernBERT-base (selected)** | Native long context (up to 8192 tokens) comfortably covers the dataset's story-length distribution without a truncation compromise — 95.3% coverage at the chosen `MAX_LENGTH=2048` (Section 4.4). Modern architectural improvements (rotary position embeddings, GeGLU activations, alternating local/global attention) offer strong per-parameter classification performance relative to classic BERT-class encoders. |

### 4.2 Why full fine-tuning, not LoRA/QLoRA
ModernBERT-base is ~150M parameters — small enough that full fine-tuning is
cheap in both time and memory, and it shares zero weights with Task 1, so
no forgetting concern applies.

### 4.3 Hyperparameters

| Parameter | Value |
|---|---|
| Model checkpoint | `answerdotai/ModernBERT-base` |
| Max sequence length | 2048 tokens |
| Batch size / grad accumulation | 4 / 2 (effective 8) |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Max epochs | 10, early stopping patience 2 |
| Metric for best model | Macro-F1 |
| Precision | FP32/default |

### 4.4 Token coverage (informed the MAX_LENGTH choice)

| Cutoff (tokens) | % of stories fully covered |
|---|---|
| 512  | 10.3% |
| 768  | 11.4% |
| 1024 | 27.0% |
| 1536 | 81.7% |
| 2048 | 95.3% |

### 4.5 Regularization ablation

Given the ~10-shot/class regime, an explicit regularization variant was
tested to check for overfitting sensitivity: `weight_decay=0.05` (vs. the
default 0.01) combined with `label_smoothing_factor=0.05` (vs. 0.0),
trained on the same fold and compared directly against the unregularized
baseline.

| Configuration | Accuracy | Macro-F1 |
|---|---|---|
| Baseline (weight_decay=0.01, no label smoothing) | 53.0% | 49.5% |
| Regularized (weight_decay=0.05, label_smoothing_factor=0.05) | 51.5% | 48.2% |

The regularized variant **did not improve** on the baseline — accuracy and
F1 both dropped slightly (−1.5 and −1.3 points respectively). This is a
legitimate negative result: the unregularized configuration was retained
for all subsequent (5-fold CV) training, and the ablation itself is useful
evidence that overfitting was not the dominant limiting factor at this
hyperparameter setting, given additional regularization did not help.

### 4.6 DeBERTa-v3-base — attempted, not completed

A single-fold DeBERTa-v3-base run (512-token context, head+tail truncation)
was attempted as a second encoder-only comparison point. It encountered a
gradient explosion early in training (loss spiking to ~7,276 before
collapsing to exactly 0.0 with NaN validation loss), producing a
chance-level result (1.0% accuracy). Given DeBERTa-v3 was a secondary,
optional comparison rather than a candidate for the final solution, and
tokenizer special-token handling was already confirmed correct as the
first diagnostic step, further root-cause investigation was not pursued
given time constraints.

---

## 4B. Qwen2.5-7B + QLoRA (Shared Instance) — Training Methodology

### 4B.1 Architecture: sharing Task 1's loaded model, not a second copy

`get_peft_model()` injects LoRA adapters into the **same** `llm_model`
object already loaded for Task 1 (4-bit NF4 quantized) — in place, not a
clone. A custom module (`QwenGenreClassifierModel`) wraps this shared
`peft_model` plus a `LayerNorm` (`pre_head_norm`) and a linear
classification head, exposing a `forward(input_ids, attention_mask,
labels)` signature compatible with HuggingFace `Trainer`.

This was verified directly, not assumed:
```python
print(id(llm_model) == id(peft_model.base_model.model))
# True
```
Confirms a single shared object graph — no second 7B copy is ever resident
in VRAM. This is also consistent with the peak combined VRAM measured
during Qwen QLoRA training (~10.21GB, Section 7), which is far below what
two resident 7B-class model instances would require.

### 4B.2 Why a classification head + LayerNorm, not generative fine-tuning
A classification head produces a direct 99-way prediction rather than free
text requiring parsing into an exact label match — more sample-efficient
and maps directly onto the required metrics. A `LayerNorm` was added before
the head after an initial diagnostic run showed loss ≈15 instead of the
expected ln(99)≈4.6 — traced to unnormalized pooled hidden states (large,
uncalibrated magnitude from a 7B model) feeding directly into the linear
head. Adding `pre_head_norm` brought initial loss to the expected range.

### 4B.3 A LoRA-specific initialization detail verified during debugging
A gradient-flow diagnostic initially flagged all `lora_A` tensors as having
zero gradient on step 1 and was mistakenly treated as a bug. This is
expected: LoRA's `lora_B` is zero-initialized by design (so the adapter
starts as a true no-op), which mathematically gates `lora_A`'s gradient to
zero on the very first backward pass. A corrected two-step diagnostic
confirmed all 224 LoRA tensors receive valid, non-zero gradients by step 2,
once `lora_B` moves off zero.

### 4B.4 Hyperparameters

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct`, 4-bit NF4 (shared with Task 1) |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj |
| Max sequence length | 2048 tokens |
| Batch size | 4 |
| Epochs | 3 (fixed, per fold) |
| Learning rate | 2e-4 |
| Weight decay | 0.01 |
| Loss | Class-weighted cross-entropy (per-fold weights, given ~10-shot imbalance) |
| Training framework | HuggingFace `Trainer`, `logging_strategy="epoch"`, `eval_strategy="epoch"` |

### 4B.5 Training time and resources

| Metric | Value |
|---|---|
| Time per fold (3 epochs, 800 train / 200 val examples) | ~107 minutes |
| Full 5-fold CV total | ~535 minutes (~8.9 hours) |
| VRAM after fold cleanup | 5.67GB allocated, 10.21GB peak |

Roughly **2x ModernBERT's per-fold training time** (~107 min vs. ~50 min) —
a real cost that persists regardless of the accuracy comparison outcome.

**Note on epoch count:** validation F1 was still climbing at epoch 3 in
every fold (e.g. fold 2: 39.1%→54.2%→62.6%), with no plateau reached within
the fixed 3-epoch budget. This suggests the reported CV numbers may
understate Qwen's ceiling — a deployment model trained for more epochs
could potentially do better still, though this was not tested given time
constraints.

---

## 5. Classification Metrics — ModernBERT (5-fold CV)

| Metric | Value |
|---|---|
| Accuracy | 52.9% ± 3.2% |
| Macro-precision | 52.8% ± 3.8% |
| Macro-recall | 52.7% ± 3.3% |
| Macro-F1 | 50.0% ± 3.5% |

| Fold | Accuracy | Precision | Recall | Macro-F1 |
|---|---|---|---|---|
| 1 | 53.0% | 51.7% | 52.8% | 49.5% |
| 2 | 50.5% | 51.4% | 50.3% | 48.3% |
| 3 | 52.0% | 51.2% | 52.0% | 48.8% |
| 4 | 59.0% | 60.3% | 58.8% | 56.8% |
| 5 | 50.0% | 49.6% | 49.7% | 46.8% |

---

## 5B. Classification Metrics — Qwen2.5-7B + QLoRA (5-fold CV)

| Metric | Value |
|---|---|
| Accuracy | **63.80% ± 0.81%** |
| Macro-precision | **62.28% ± 2.25%** |
| Macro-recall | **63.74% ± 0.87%** |
| Macro-F1 | **60.41% ± 1.21%** |

| Fold | Accuracy | Precision | Recall | Macro-F1 |
|---|---|---|---|---|
| 1 | 63.50% | 60.67% | 63.38% | 59.32% |
| 2 | 65.00% | 66.53% | 64.90% | 62.57% |
| 3 | 64.50% | 62.61% | 64.65% | 60.88% |
| 4 | 63.00% | 60.93% | 62.88% | 59.67% |
| 5 | 63.00% | 60.66% | 62.88% | 59.61% |

Notably tighter variance across folds (±0.81% accuracy) than ModernBERT
(±3.23%) — Qwen QLoRA is not only more accurate on average but more
*consistent* fold-to-fold.

**Confusion matrix / error analysis:** the original five-fold checkpoints
saved the LoRA adapter and classification head but omitted
`pre_head_norm`'s trained affine parameters. Reloading one of those
incomplete checkpoints with a newly initialized normalization layer produced
only 15.5% accuracy instead of the original fold's 63.0%, so that
reconstruction was rejected as invalid evidence. The original five-fold
accuracy and F1 values remain valid because they were computed in memory
during the actual training runs.

To correct the checkpointing problem, fold 0 was retrained and all three
trainable components were saved: the LoRA adapter, classification head, and
`pre_head_norm`. The retrained run achieved **59.5% accuracy and 55.8%
macro-F1**. Reloading the complete checkpoint reproduced the same **59.5%
accuracy**, confirming that the saved classifier was reconstructable. This
corrected fold-0 run was used to generate the Qwen confusion matrix and
per-class classification report. It is reported as a separate diagnostic
run and is not substituted into the original five-fold CV results above.

---

## 6. Comparison and Statistical Significance

### 6.1 Head-to-head

| Approach | Accuracy | Macro-F1 | Params trained | Train time/fold | Forgetting risk |
|---|---|---|---|---|---|
| TF-IDF + SVM (baseline) | 44.5% | 41.3% | — | seconds | none (separate) |
| Frozen BGE-small + LogReg (baseline) | 45.0% | 39.4% | ~1K | seconds | none (separate) |
| ModernBERT-base (full fine-tune) | 52.9% ± 3.2% | 50.0% ± 3.5% | ~150M | ~50 min | none (separate model) |
| **Qwen2.5-7B + QLoRA (shared instance)** | **63.8% ± 0.8%** | **60.4% ± 1.2%** | tens of millions | ~107 min | Structurally near-zero (Section 8) |

### 6.2 Fold-range comparison
Qwen's **worst** fold (63.0%) exceeds ModernBERT's **best** fold (59.0%) —
zero overlap between the two models' fold-accuracy ranges.

### 6.3 Paired statistical significance
Both models were evaluated on identical fold splits, enabling a paired
comparison (more statistically powerful than comparing means alone at n=5):

| Test | Accuracy | Macro-F1 |
|---|---|---|
| Paired t-test | t=5.92, **p=0.0041** | t=5.16, **p=0.0067** |
| Wilcoxon signed-rank | p=0.0625 | p=0.0625 |
| Cohen's d (paired, effect size) | 2.65 | 2.31 |

The paired t-test shows significance at p<0.01 for both metrics, with a
very large effect size (d>2 in both cases; d>0.8 is conventionally "large").
The Wilcoxon test lands just above the conventional 0.05 threshold — this
is a known limitation at n=5 pairs, where Wilcoxon's minimum achievable
p-value is 0.0625 regardless of effect size (2^5=32 possible sign
arrangements). Both tests are reported rather than selecting the
significant one, for an honest picture of the evidence.

**Conclusion: the accuracy advantage is real and substantial, not
attributable to chance or fold-selection luck.**

---

## 7. Computational Resources: VRAM

| Component | Precision | Memory |
|---|---|---|
| Qwen2.5-7B (Task 1 chatbot, isolated) | 4-bit NF4 | 5.32GB allocated, 5.44GB peak |
| ModernBERT-base (Task 2) | FP32/default | ~0.6GB serialized weights |
| Combined (Task 1 resident + ModernBERT training) | — | 17.33GB peak (measured) |
| Combined (Task 1 resident + Qwen QLoRA training, shared instance) | — | ~10.21GB peak (measured, post-fold-cleanup) |

Both architectures comfortably fit the 24GB combined budget. The
shared-instance Qwen approach has a **lower** combined peak than the
two-model ModernBERT approach, since it never holds two separate full model
graphs in memory simultaneously.

---

## 8. Multi-Task Preservation (Qwen QLoRA — Shared Instance)

Because Qwen QLoRA shares Task 1's model weights directly, forgetting
prevention required explicit proof, built in three layers of increasing
strength:

### 8.1 Structural proof
```python
base_param_names_in_optimizer = [
    name for name, param in peft_model.named_parameters()
    if param.requires_grad and "lora_" not in name
]
# Result: 0 — confirmed across all 5 CV folds
```
Zero non-LoRA parameters were ever in the trainable parameter set. Base
model weights are **mathematically guaranteed unchanged** — this holds
regardless of any generation-based test, since it's a property of what was
passed to the optimizer, not an empirical observation.

### 8.2 A real bug found and fixed during verification
An initial post-training generation test (adapter disabled) produced
degenerate, repetitive output (e.g. repeated underscore/digit tokens)
rather than coherent text. Root cause: the model was left in **training
configuration** — gradient checkpointing still enabled and `use_cache=False`
— neither of which was reset before attempting generation. This is a known
failure mode (gradient checkpointing interacting badly with KV-cache-based
generation) and produced clearly broken output, not a subtle one. Fixed by
explicitly calling `.eval()`, `gradient_checkpointing_disable()`, and
resetting `use_cache=True` before generation. This is a genuine production
readiness finding: any system that toggles between training and serving
states needs explicit state-reset handling, not an implicit assumption
that training cleanup happens automatically.

### 8.3 Functional / true-baseline proof
A genuine pre-training baseline was captured in a **fresh runtime**, before
any Task 2 or LoRA code executed (rather than only reconstructing a
baseline post-hoc by disabling an already-attached adapter). Deterministic
(greedy, `do_sample=False`) generation was used throughout, since sampling
would make any comparison meaningless regardless of actual forgetting.

A trained fold's LoRA adapter was then loaded onto this same fresh
`llm_model` instance and disabled (`peft_model.disable_adapter_layers()`).
Per PEFT's implementation, this **bypasses LoRA computation entirely**
rather than zeroing a learned delta — computationally identical to a model
that never had LoRA injected, not an approximation of one.

**Result:** with the adapter disabled, deterministic generation on a fixed
query set matched the true baseline. Re-enabling the adapter and comparing
against the baseline confirmed the toggle mechanism is a genuine, working
switch (output changes when engaged) rather than a silent no-op.

### 8.4 Summary
| Layer | What it proves | Result |
|---|---|---|
| Structural (8.1) | Base weights could not have changed | ✅ 0 non-LoRA trainable params |
| Functional (8.3) | Disabling the adapter reproduces the true pre-training baseline | ✅ Identical under deterministic decoding |
| Toggle integrity (8.3) | The adapter mechanism is a real, working switch | ✅ Output differs when adapter enabled |

The structural proof (8.1) is the load-bearing evidence — it is a
guarantee, not an observation. The functional checks (8.3) are valuable
confirmatory evidence and caught a real implementation bug (8.2) that the
structural proof alone would not have surfaced, but they are not the
primary basis for the no-forgetting claim.

---

## 9. Challenges and Reflections

- **Sequence length vs. coverage trade-off** required empirical measurement
  rather than assumption (Section 4.4).
- **Regularization ablation (Section 4.5)** tested whether the ~10-shot/class
  regime needed stronger regularization than the default; it did not — a
  useful negative result rather than an unexplored gap.
- **LayerNorm-before-head fix (Section 4B.2):** caught via a numerical
  sanity check (initial loss ≈15 vs. expected ln(99)≈4.6) rather than
  assumed correct because the code ran without error.
- **LoRA zero-initialization gradient gotcha (Section 4B.3):** a
  false-positive bug flag corrected by understanding *why* LoRA is
  initialized the way it is, not just how to call the library.
- **Gradient-checkpointing/generation-config bug (Section 8.2):** training
  state (checkpointing enabled, `use_cache=False`) silently persisted into
  generation, producing degenerate output that could easily have been
  misread as evidence of forgetting rather than a state-management bug.
- **Checkpointing gap for `pre_head_norm` (Section 5B):** a real lesson —
  when checkpointing a custom multi-component model, every trainable
  submodule needs explicit saving, not just the "obvious" ones (adapter,
  head). A 15.5%-vs-63.0% reconstruction gap exposed the omission. The
  pipeline was then corrected by retraining fold 0 and saving the adapter,
  head, and normalization layer; the reloaded checkpoint exactly reproduced
  that retrain's 59.5% accuracy.
- **What would be done differently given more time:** checkpoint all
  submodules from the start; run a full deployment retrain for Qwen; test
  DeBERTa-v3 and potentially a data-augmentation approach given the
  ~10-shot/class regime.
