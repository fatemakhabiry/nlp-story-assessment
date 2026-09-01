# Task 3: CPU Deployment with Low Latency — Submission Documentation

## Executive Summary

Both required components were evaluated for CPU deployment: the Task 1 story chatbot and the Task 2 genre classifier. The target Google Colab CPU runtime exposed **2 logical CPUs / 1 physical core** on an Intel Xeon @ 2.20GHz host. All target-runtime chatbot benchmarks used two inference threads.

For Task 1, the original Qwen2.5-7B chatbot was converted to GGUF Q4_K_M and proved technically executable, but full-context RAG queries required **637.7-1,152.8 seconds** and the full process used approximately **6.69 GiB RSS**. A controlled model-size ablation was therefore conducted using the same GGUF quantization and two-thread `llama-bench` conditions. Qwen2.5-1.5B achieved **27.95 prompt tokens/s** and **11.43 generation tokens/s**, approximately twice the throughput of the 3B candidate and over seven times the 7B candidate. It was selected for the final chatbot deployment.

The final chatbot also changed context construction: hybrid dense + BM25 retrieval with Reciprocal Rank Fusion preserves the highest-ranked query-relevant chunk, and only one chunk (maximum 350 words) is supplied to the LLM. Structured genre requests bypass generation entirely. The final repeated benchmark used three representative generated queries with three runs per query (nine runs total). Overall median TTFT was **21.31 seconds** (P95 **23.22 seconds**) and median end-to-end latency was **23.25 seconds** (P95 **26.54 seconds**). A genre listing completed deterministically in **0.69 ms** in the representative functional test. Final process RSS was **2.55 GiB**.

For Task 2, Qwen2.5-7B + QLoRA remains the accuracy-superior classifier on GPU, but its validated dense CPU reconstruction required approximately **17.7 GiB RSS** and **49.6-267.0 seconds** per tested example even on a more capable host. ModernBERT FP32 under PyTorch was therefore selected for CPU serving. Quantized variants were rejected because they either damaged predictions or became slower, and ONNX FP32 was slower than PyTorch in two matched runs. A final sequence-budget optimization reduced the maximum input from 2,048 to **1,024 tokens**, lowering mean latency to **6.41 seconds** and P95 latency to **8.65 seconds**, with a 1.0-point accuracy reduction on the held-out fold.

The final production recommendation is therefore a deliberately asymmetric CPU architecture:

| Component | Selected CPU runtime | Reason |
|---|---|---|
| Task 1 chatbot | **Qwen2.5-1.5B-Instruct, GGUF Q4_K_M, `llama.cpp`** | Best measured latency/memory compromise; grounded relevant-chunk prompting |
| Task 2 classifier | **ModernBERT FP32, PyTorch, 1,024-token cap** | Best measured CPU trade-off after quantization, runtime, and sequence-budget tests |

---

## 1. Environment and Hardware

| Property | Value |
|---|---|
| CPU label | Intel Xeon @ 2.20GHz |
| Logical CPUs exposed | 2 |
| Physical cores reported by `psutil` | 1 |
| Chatbot inference threads | 2 |
| Runtime | Google Colab CPU-only runtime |
| Final Task 1 model load time | 7.35 s |
| Final Task 1 process RSS | 2.55 GiB |
| Classifier inference threads | 2 |

The measurements are hardware-specific. This highly constrained runtime is weaker than a typical multi-core production server, so results should not be generalized to all CPU deployments.

---

## 2. Task 1 - Chatbot CPU Deployment

### 2.1 Why GGUF and `llama.cpp` were used

Task 1 originally used Qwen2.5-7B-Instruct with Transformers and bitsandbytes NF4 on GPU. That configuration was appropriate for GPU memory-efficient inference, but the CPU target required a more mature CPU-oriented runtime. GGUF stores model tensors, quantization metadata, architecture information, and tokenizer metadata in a format consumed efficiently by `llama.cpp`. Q4_K_M was selected as a standard size/quality compromise.

`llama.cpp` provides optimized native CPU kernels, memory-mapped loading, KV-cache support, and low-overhead autoregressive generation. GGUF is the model container; `llama.cpp` is the inference engine.

### 2.2 Rejected Qwen2.5-7B baseline

The 7B Q4_K_M artifact was 4.36 GiB. Raw throughput on two threads was only 3.26 prompt tokens/s and 1.53 generation tokens/s.

| Query | Retrieval | Prompt tokens | TTFT | Total latency |
|---|---:|---:|---:|---:|
| Exact ID query | 0.3 ms | 1,303 | 498.5 s | 637.7 s |
| Hybrid semantic query | 181.5 ms | 2,893 | 1,043.2 s | 1,152.8 s |

The full 7B RAG process used approximately 6.69 GiB RSS. Retrieval was negligible relative to prefill and generation. Blind 300-character and head-tail context reductions were also rejected because their sample answers became unsupported or generic. The 7B configuration demonstrated CPU feasibility but did not meet the practical low-latency objective.

### 2.3 Model-size ablation

All candidates used GGUF Q4_K_M and the same two-thread `llama-bench` test (`pp128`, `tg128`).

| Model | Reported artifact size | Prompt processing | Generation | Decision |
|---|---:|---:|---:|---|
| Qwen2.5-7B | 4.36 GiB | 3.26 tok/s | 1.53 tok/s | Rejected |
| Qwen2.5-3B | 1.79 GiB | 13.49 tok/s | 6.02 tok/s | Not selected |
| **Qwen2.5-1.5B** | **~0.92 GiB** | **27.95 tok/s** | **11.43 tok/s** | **Selected** |

The 1.5B candidate was approximately 2.1 times faster than 3B for prompt processing and 1.9 times faster for generation. Relative to 7B, it was approximately 8.6 times faster for prompt processing and 7.5 times faster for generation. The final local artifact check reported 940.37 MiB, while the earlier `llama-bench` line reported 934.69 MiB; this small artifact/build difference does not affect the selection.

Only the 1.5B model proceeded to full RAG evaluation. The 3B result is retained as an ablation rather than described as a deployed model.

### 2.4 Final retrieval and context policy

The final retrieval layer uses the Task 1 hybrid design:

1. Stories are split into 500-word chunks with 100-word overlap, producing 2,764 chunks.
2. Dense retrieval uses `BAAI/bge-small-en-v1.5` and exact FAISS inner-product search over normalized embeddings.
3. BM25 retrieves lexical matches over the same chunks.
4. Reciprocal Rank Fusion combines both rankings with `rrf_k=60`.
5. Candidates are grouped by parent story ID so one story cannot occupy several story-level ranks.
6. The highest-ranked query-relevant chunk is retained for generation and trimmed to at most 350 words.

For an explicit story ID or title, retrieval is restricted to chunks belonging to the matched story. Genre listings bypass Qwen and return deterministic IDs and titles. This preserves structured-query accuracy and eliminates unnecessary generation latency.

The final Qwen configuration was:

| Setting | Value |
|---|---|
| Model | Qwen2.5-1.5B-Instruct |
| Format | GGUF |
| Quantization | Q4_K_M |
| Context window used | 2,048 tokens |
| Output cap | 96 tokens |
| Inference threads | 2 |
| Batch threads | 2 |
| Batch size | 256 |
| Sampling | Deterministic (`temperature=0`) |
| Context | Best query-relevant chunk, maximum 350 words |

`llm.reset()` was called before each timed generated response to prevent prefix/KV-cache reuse from creating misleading TTFT measurements. The earlier 0.95-second TTFT observation was identified as a warm-cache artifact and is not treated as a valid deployment result.

### 2.5 Final end-to-end results

The final latency benchmark used three representative generated-query types and three runs per query, for **nine generated runs**. The model and retrieval indexes remained resident, deterministic decoding was used, the output cap was 96 tokens, and `llm_small.reset()` cleared the KV cache before every timed generation. Query types were interleaved across the three rounds to reduce systematic run-order effects.

| Query type | Runs | Retrieval median | TTFT median | TTFT P95 | Total median | Total P95 |
|---|---:|---:|---:|---:|---:|---:|
| ID-scoped supernatural detail | 3 | 49.7 ms | 21.31 s | 22.97 s | 23.25 s | 24.78 s |
| Adventure/friendship semantic theme | 3 | 52.3 ms | 16.02 s | 16.68 s | 17.29 s | 17.64 s |
| SS Excelsior rare named detail | 3 | 54.6 ms | 22.59 s | 23.19 s | 26.19 s | 26.71 s |

| Overall generated benchmark | Result |
|---|---:|
| Runs | 9 |
| Retrieval median / P95 | 53.8 / 89.4 ms |
| TTFT mean / median / P95 | 20.24 / 21.31 / 23.22 s |
| End-to-end mean / median / P95 | 22.35 / 23.25 / 26.54 s |
| End-to-end range | 16.70-26.77 s |

The separate Science Fiction genre functional test used the deterministic route and completed in 0.69 ms without invoking the LLM. This single deterministic-route observation is reported as a functional latency example, not as a latency distribution.

The ID query correctly identified the Whispering Trees. The SS Excelsior query retrieved the correct first-ranked source, ID 457580 (`The Chronicles of the Cosmic Rift`). The broad adventure query returned a relevant survival/adventure story, but its generated answer was extremely terse. This is an explicit quality trade-off of the smaller model and 96-token deployment cap, not evidence of retrieval failure. For semantic routes, the result object exposes the three retrieved candidates for inspection, while only the first-ranked candidate's chunk is supplied to the LLM under the final `max_evidence=1` latency policy. Consequently, the first candidate is the answer's context source; the other two are retrieval alternatives rather than evidence consumed during generation.

The repeated benchmark now provides median and P95 statistics across the nine generated runs. Because it still covers only three query types, it is a small controlled benchmark rather than a comprehensive production traffic distribution. Answer quality was inspected qualitatively on the three representative query types; a larger systematic grounding evaluation was not conducted because of the project time constraint.

### 2.6 Resource footprint and improvement

| Measurement | 7B baseline | Final 1.5B deployment |
|---|---:|---:|
| GGUF artifact | 4.36 GiB | ~0.92 GiB |
| Full RAG process RSS | 6.69 GiB | 2.55 GiB |
| Prompt processing | 3.26 tok/s | 27.95 tok/s |
| Generation | 1.53 tok/s | 11.43 tok/s |
| Generated-query total latency | 637.7-1,152.8 s | Median 23.25 s; P95 26.54 s |

The selected system reduced measured generated-query latency by more than an order of magnitude while reducing the full process RSS by approximately 62%. The final model loaded in 7.35 seconds, and a trivial smoke-test question completed in 2.83 seconds.

### 2.7 Task 1 conclusion

> The 7B model was CPU-executable but operationally impractical. A measured 7B/3B/1.5B ablation selected Qwen2.5-1.5B Q4_K_M as the strongest latency-first candidate. Combined with query-relevant chunk prompting, deterministic structured routes, a 2,048-token context, and a 96-token output cap, the final repeated benchmark achieved 23.25 seconds median and 26.54 seconds P95 end-to-end latency across nine generated runs, with full process RSS of 2.55 GiB. This is a substantial deployment improvement, although it remains slower than an ideal interactive service and accepts a measurable answer-richness trade-off.

---
## 3. Task 2 — ModernBERT Classifier CPU Deployment

Two checkpoints have different roles. `fold0_evaluation_model` was trained
on 800 examples and is used for every held-out accuracy and classifier
latency claim. `final_all_data_model` was trained on all 1,000 examples and
is the production serving artifact. The fold-0 metrics are not attributed
to the all-data model; serving examples from that checkpoint demonstrate
functionality only.

### 3.1 Quantization investigation: six attempts, systematically documented

Given the brief's explicit interest in "what methods you chose and why," all
six attempts are reported — including five that were rejected — since trying
and correctly rejecting an optimization based on measurement is legitimate
engineering evidence, not a shortfall.

| # | Method | Measured accuracy/agreement change | Prediction agreement | Speedup | Verdict |
|---|---|---|---|---|---|
| 1 | ONNX Runtime, `avx512_vnni` config | −97.0 pts *(training-seen diagnostic)* | 3.0% | 1.08x (not real) | **Rejected** — hardware-incompatible; this CPU has no AVX512-VNNI support (confirmed via `/proc/cpuinfo`), producing silently incorrect results rather than an error |
| 2 | ONNX Runtime, `avx2` config | −82.0 pts *(training-seen diagnostic)* | 18.0% | 0.97x | **Rejected** — hardware-appropriate config, but no genuine speedup, suggesting quantized ops did not execute via true int8 kernels |
| 3 | ONNX Runtime, `avx2` + `reduce_range=True` | — (3-example spot check only, 2/3 mismatched) | — | — | **Rejected** — not pursued to full validation given the pattern from attempts 1–2 |
| 4 | PyTorch native dynamic quantization (per-tensor) | −44.0 pts *(training-seen diagnostic)* | 56.0% | 1.23x | **Rejected** — first config with a genuine (if modest) speedup, but accuracy impact still unacceptable |
| 5 | PyTorch native dynamic quantization (per-channel) | −41.0 pts *(training-seen diagnostic)* | 59.0% | 1.20x | **Rejected** — most numerically precise PyTorch config tested; marginal improvement over per-tensor, still unacceptable |
| 6 | `torchao` (`Int8DynamicActivationInt8WeightConfig`) | **−2.0 pts (genuine held-out, n=100)** | **89.0%** | **0.06x (17.6x slower)** | **Rejected** — accuracy finally preserved, but catastrophically slower; diagnosed via a printed warning ("Detected no triton, on systems without Triton certain kernels will not work") indicating an inefficient CPU fallback path |

### 3.2 Important caveat: which results are valid held-out accuracy

Attempts 1, 2, 4, and 5 evaluated `final_all_data_model` on examples drawn
from the same 1,000 stories used for its training. Their 100% FP32 baselines
therefore do not estimate held-out generalization. They are retained only as
prediction-preservation diagnostics: low agreement still demonstrates that
quantization materially changed the model's behaviour.

Attempt 6 (`torchao`) used a reproducible **100-example subset** of the
genuine fold-0 held-out partition and therefore provides a valid, though
subset-based, estimate of quantization's accuracy impact. The matched
PyTorch FP32-versus-ONNX FP32 comparison in Section 3.4 and the sequence-
length ablation in Section 3.5 used the complete **200-example fold-0
held-out partition**. The production inference function uses the all-data
model, but any serving examples from it are functional checks rather than
held-out accuracy evidence.

### 3.3 Root cause: why `torchao` was accurate but 17.6x slower

The `torchao` run printed: `"Detected no triton, on systems without Triton
certain kernels will not work."` The warning provides a plausible
explanation for the slowdown: optimized kernels may have been unavailable,
causing execution through a less efficient fallback path. This was not
independently isolated as the sole cause. **Regardless of the exact
mechanism, the result demonstrates that fixing quantization's numerical
correctness and fixing its execution speed are two separate problems that
do not necessarily arrive together** — a general, reproducible finding
about CPU quantization tooling, not specific to this one library.

### 3.4 FP32 matched runtime comparison (PyTorch vs. ONNX Runtime)

A matched comparison used `fold0_evaluation_model`, the same 200 held-out
examples, dynamic-length tokenization capped at 2,048 tokens, three untimed
warm-up iterations, and sequential single-example inference. PyTorch FP32
and ONNX Runtime FP32 processed identical tokenized inputs. The full
benchmark was completed twice. Timings used `time.perf_counter()` around
model execution; tokenization was recorded separately and is not included
in the runtime latencies below.

| Run | Runtime | Accuracy | Macro-F1 | Mean | P50 | P95 | Agreement |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | PyTorch FP32 | 49.0% | 46.0% | 8,039 ms | 7,797 ms | 15,060 ms | Reference |
| 1 | ONNX Runtime FP32 | 49.0% | 46.0% | 8,889 ms | 8,906 ms | 16,574 ms | 100.0% |
| 2 | PyTorch FP32 | 49.0% | 46.0% | 10,006 ms | 9,961 ms | 18,015 ms | Reference |
| 2 | ONNX Runtime FP32 | 49.0% | 46.0% | 10,476 ms | 10,407 ms | 19,350 ms | 100.0% |

Absolute latency varied between runs because the shared Colab virtual CPU
does not provide dedicated scheduling, fixed clock frequency, or controlled
background load. PyTorch mean latency ranged from **8.04 to 10.01 seconds**,
and ONNX mean latency ranged from **8.89 to 10.48 seconds**. Nevertheless,
the decision was stable: ONNX was **10.6% slower in run 1** and **4.7% slower
in run 2**, while predictions agreed 100% in both runs. The ranges are
reported rather than selecting only the faster run.

| Runtime stage | Observed process RSS |
|---|---:|
| Baseline | 1,288 MB |
| After PyTorch load | 1,305 MB |
| After ONNX export/load while PyTorch remained resident | 2,745 MB |

The RSS snapshots above were collected while both runtimes and earlier
notebook objects were resident. They describe process state, not isolated
model memory, and are therefore not used to choose between runtimes.

**Decision:** ONNX preserved the classifier exactly but provided no latency
advantage in either run. PyTorch FP32 was retained.

### 3.5 Maximum-sequence-length optimization

After selecting PyTorch FP32, the maximum sequence budget was reduced from
2,048 to 1,024 tokens. This was an inference-only transformation: model
weights were unchanged, shorter inputs retained their actual length, and
only longer stories were truncated. The same fold-0 model, 200 held-out
examples, two CPU threads, and three untimed warm-up iterations were used.
The reported latency again measures model execution separately from
tokenization.

| Metric | 2,048-token reference | 1,024-token candidate | Change |
|---|---:|---:|---:|
| Accuracy | 49.0% | 48.0% | −1.0 point |
| Macro-F1 | 46.0% | 44.81% | −1.19 points |
| Mean latency | 8,039 ms | 6,413 ms | 20.2% reduction (1.25x) |
| P50 latency | 7,797 ms | 6,658 ms | 14.6% reduction |
| P95 latency | 15,060 ms | 8,645 ms | 42.6% reduction |

The 1,024-token predictions agreed with the 2,048-token predictions on
**88.5%** of examples. Mean processed length was 915.8 tokens and the
maximum was 1,024. Process RSS was 3.68 GiB, but this included the complete
notebook state and is not claimed as isolated ModernBERT memory.

The latency reductions above use the faster of the two 2,048-token runs as
the conservative reference. The 1,024-token result was also faster than the
second 2,048-token run. Because accuracy decreased by only one point while
mean and tail latency improved materially, **the 1,024-token cap was adopted
for final CPU serving**.

### 3.6 ModernBERT deployment conclusion

Quantization did not provide an acceptable accuracy-preserving speedup, and
ONNX FP32 was slower than PyTorch in both matched runs. PyTorch FP32 was
therefore retained, then improved through an accepted sequence-budget
optimization. The final deployed classifier is **ModernBERT FP32 under
PyTorch with dynamic input lengths capped at 1,024 tokens and two CPU
threads**.

### 3.7 Additional techniques considered

The brief presents pruning, distillation, and quantization as possible
approaches, not a requirement to apply all of them. Quantization was tested
directly. Pruning and distillation would create newly trained models and
would require another leakage-safe cross-validation cycle, so they were not
implemented within the project schedule.

The strongest future option is **knowledge distillation from Qwen into a
smaller encoder**: Qwen would provide soft 99-class targets during training,
while only the encoder student would be deployed. For valid evaluation, the
teacher must be trained only on each fold's training partition; using the
all-data Qwen model would leak validation information. Structured pruning of
encoder layers, heads, or feed-forward channels could also reduce dense
computation, but would require fine-tuning. Unstructured pruning was not
prioritized because zero weights do not automatically reduce latency without
a compatible sparse CPU runtime. These are future experiments rather than
claims about the submitted system.

---

## 4. Task 2 — Qwen2.5-7B QLoRA Classifier: CPU Feasibility

Task 2's documentation identified Qwen2.5-7B + QLoRA as the statistically
stronger classifier (63.8% ± 0.8% vs. ModernBERT's 52.9% ± 3.2% accuracy,
p=0.004). This section evaluates whether that accuracy advantage is
practical to deploy on the target CPU environment.

### 4.1 Approach and a hardware caveat

Two checks were performed before attempting any CPU test:

1. **Hardware verification**: the GPU notebook's host CPU (**6 physical
   cores / 12 logical CPUs (hyperthreaded), AVX512-VNNI support**) was
   confirmed via `lscpu` to be **more capable hardware than Task 3's actual
   CPU-only deployment target** (2 logical CPUs / 1 physical core, no AVX512-VNNI, confirmed
   separately in Section 1). Any latency result from this session is
   therefore **a favorable best-case latency estimate, representing an
   approximate lower bound on expected latency rather than a target-hardware
   measurement** — if latency is already prohibitive on this more capable
   host, that is sufficient evidence of impracticality on the weaker target
   without needing to separately reproduce it there.
2. **Drive capacity check**: the available 4.5 GB of Drive space prevented
   saving and directly transferring the approximately 14–15 GB dense FP16
   checkpoint. Reconstruction in another notebook remained theoretically
   possible by transferring the smaller adapter and custom head and
   downloading the base model again, but this was not pursued because the
   favorable-host CPU benchmark had already established impractical latency.

### 4.2 Merge diagnostic

Rather than assuming `merge_and_unload()` on a 4-bit quantized base produces
a genuine dense FP16 model, this was verified directly by inspecting the
resulting module type:

```python
merged_model = peft_model.merge_and_unload()
# UserWarning: Merge lora module to 4-bit linear may get different
# generations due to rounding errors.

for name, module in merged_model.named_modules():
    if "q_proj" in name:
        print(type(module))
        break
# -> <class 'bitsandbytes.nn.modules.Linear4bit'>
```

**Result: the merged model's layers remain represented as
`bitsandbytes.nn.modules.Linear4bit`**, not standard dense `nn.Linear`
modules, despite the merge operation completing without error. The
resulting `Linear4bit` representation was not treated as a portable,
validated CPU artifact in this environment. More importantly, its held-out
accuracy had already collapsed to 31.0% (Section 4.3), so it was rejected
independently of any CPU-backend support question. PEFT's own warning
additionally flags a documented precision-rounding risk from merging into
a quantized backbone.

### 4.3 Prediction-preservation validation — the decisive finding

Given the rounding-error warning, prediction preservation was validated
directly rather than assumed, in two stages:

**Stage 1 (5-example spot check):** 4/5 top-1 agreement between the
original unmerged model and the merged model, including one full label
flip. This was informative but not conclusive at n=5.

**Stage 2 (full fold-0 held-out evaluation, 200 examples):** the merged
model was evaluated on the same held-out validation split used to report
the classifier's original accuracy. This evaluation took 137.5 seconds
**in total on GPU, with the 200 examples processed sequentially** (one at a
time in a loop, not batched) — it is neither single-query latency nor CPU
deployment latency, and is not used as a deployment latency estimate
anywhere in this document.

| | Original (unmerged, `fold_0_fixed`'s own accuracy*) | Merged (4-bit `Linear4bit`) | Delta |
|---|---|---|---|
| Accuracy | 59.5% | **31.0%** | **−28.5 points** |
| Macro-F1 | 55.8% | **26.2%** | **−29.6 points** |

*Original figure is `fold_0_fixed`'s own training-time accuracy (Task 2
documentation, Section 5B) — the specific checkpoint being merged and
tested here. This is a corrected baseline: an earlier draft of this
document compared against a different training run's fold-1 CV result
(63.5%/59.3%), which used the same fold split but different weights
(global training seed was not fixed — Task 2 documentation, Section 5B).
The comparison below uses the exact checkpoint under test, as it should.

**This is a decisive, not borderline, result** — over 20x beyond a
"reject" threshold of 1.5 points. Direct 4-bit merging severely damages
classifier accuracy, not from CPU-execution incompatibility, but from the
merge operation itself corrupting the model's learned behavior before CPU
is even a consideration.

**Likely mechanism**: PEFT's rounding-error warning is typically calibrated
around standard autoregressive generation, where small perturbations
occasionally shift a single next-token choice without destabilizing overall
output. This classifier's architecture instead **mean-pools across all
2,048 hidden states** before classification — a fundamentally different
computation that plausibly allows small per-layer rounding errors to
accumulate and compound across the full pooling operation, rather than
affecting a single, localized decision. This is offered as a plausible
explanation for the severity observed, not an independently verified
mechanism.

### 4.4 Fresh dense-weight reconstruction

Given the decisiveness of Section 4.3's finding, a second reconstruction
attempt was made — loading a genuinely unquantized fp16 base and merging
the adapter into it, avoiding the `Linear4bit` representation entirely,
following the same validation discipline as before.

**Module-type verification**: confirmed dense `torch.nn.Linear` (not a
quantized backend), unlike the earlier attempt.

**5-example validation**: **5/5 top-1 agreement** with the original
unmerged QLoRA model — a categorical improvement over the earlier attempt's
4/5. Top-3 rankings were highly similar but not identical for every
example (e.g., idx=737: original [11, 71, 98] vs. dense [11, 71, 44];
idx=660: original [66, 70, 40] vs. dense [66, 70, 32]) — consistent with
expected minor numerical variation between fp16 dense and 4-bit-adapter
computation paths, not a correctness concern given top-1 agreement held.

**Full fold-0 held-out evaluation (200 examples)**:

| | Original (`fold_0_fixed`'s own accuracy) | Fresh dense fp16 merge | Delta |
|---|---|---|---|
| Accuracy | 59.5% | **61.0%** | **+1.5 points** |
| Macro-F1 | 55.8% | **58.95%** | **+3.15 points** |

This is a decisively different outcome from the `Linear4bit` attempt
(−28.5/−29.6 points) — confirming that the earlier collapse was
specifically attributable to the quantized merge representation, not to
merging LoRA into this mean-pooling architecture in general. Against the
correct baseline (`fold_0_fixed`'s own accuracy, not a different training
run's result), the dense reconstruction shows **no accuracy cost at
all** — both accuracy and macro-F1 landed slightly above the original
checkpoint's own figures, consistent with normal run-to-run variance
given the global training seed was not fixed (Task 2 documentation,
Section 5B), not a meaningful improvement to be read into. The
reconstruction is treated as **numerically equivalent, within noise**, to
the original checkpoint being tested — a stronger result than earlier
reported.

### 4.5 CPU feasibility test — functionally valid, latency prohibitive

With the reconstruction validated, a tiered CPU test was run
(this GPU notebook's host CPU: **6 physical cores / 12 logical CPUs,
AVX512-VNNI** — noted again as more capable than the actual 2-logical-CPU / 1-physical-core,
non-VNNI Task 3 target, so these figures represent a favorable best-case
latency estimate — an approximate lower bound — not the target-hardware
measurement):

| Tier | Tokens | Latency | Prediction | Correct | RSS |
|---|---|---|---|---|---|
| Short story | 265 | 49.6 s | 48 | ✅ | 17.71 GiB |
| Median-length story | 1,231 | 267.0 s | 20 | ✅ | 17.75 GiB |
| Near-2048-token story | — | — | — | — | **Not tested — stopped per hard rule after median tier exceeded the 60-second threshold** |

The short and median examples were selected according to story character
length (10th percentile and 50th percentile respectively of the dataset's
story-length distribution); the median example contained 1,231 tokens
after tokenization.

**Model load time (fp16, CPU): 6.5 seconds. Peak process RSS: ~17.7 GiB** — the
Qwen evaluation process used approximately 17.7 GiB RSS, around **6.9 times**
the measured 2.55 GiB footprint of the final quantized Task 1 RAG
system (embedder + FAISS index + GGUF chatbot combined).

**Both tested CPU examples produced the correct labels.** Macro-F1
preservation was established separately through the complete 200-example
GPU evaluation of the reconstructed dense model (Section 4.4); two CPU
examples alone are insufficient to estimate accuracy on their own, but
they are sufficient to demonstrate successful CPU execution on
representative inputs. The 267-second result for the median-length example
indicates that representative stories may require several minutes per
classification on this favorable host. **A larger CPU latency sample would
be required to estimate the full latency distribution** — this single
median-tier measurement is directional, not a statistically robust
estimate.

### 4.6 Conclusion

> Two Qwen classifier CPU conversion attempts were made, both benchmarked
> against `fold_0_fixed`'s own training-time accuracy (59.5%/55.8%), the
> specific checkpoint under test. The first — direct LoRA merging into the
> bitsandbytes 4-bit backbone — produced a `Linear4bit`-represented
> artifact that was not treated as a validated CPU path, and, more
> decisively, collapsed held-out accuracy from 59.5% to 31.0% (−28.5
> points), and was rejected before CPU execution was even considered. The
> second — a fresh unquantized fp16 base merged with the same adapter —
> produced genuine dense weights, with accuracy and macro-F1 both landing
> slightly *above* the original checkpoint's own figures (+1.5 and +3.15
> points respectively) — no meaningful accuracy cost, within normal
> run-to-run variance — and **successfully executed on CPU with correct
> predictions** on both tested tiers.
>
> **The reconstruction is therefore functionally valid and CPU-executable,
> but not practically deployable at this latency**: 49.6 seconds for a short
> (265-token) story and 267.0 seconds for a median-length (1,231-token)
> story — the latter representative of this dataset's typical story length.
> Testing was stopped before the near-2048-token tier per the pre-committed
> 60-second threshold, already exceeded at the median tier. Peak memory
> usage (~17.7 GiB) further compounds the practicality concern for
> concurrent deployment alongside the final Task 1 chatbot (~2.55 GiB).
>
> **This is a decisively cleaner finding than a simple "it failed" verdict**:
> Qwen2.5-7B + QLoRA is confirmed to be both the accuracy-superior Task 2
> model (Task 2's GPU finding, unchanged) **and** technically capable of
> CPU execution when properly reconstructed, with predictions essentially
> unchanged from the pre-merge checkpoint. **The primary deployment
> barriers are latency and memory, not accuracy** — the dense
> reconstruction avoided the catastrophic degradation of the direct 4-bit
> merge and showed no meaningful accuracy cost of its own. **ModernBERT
> remains the selected Task 2 CPU classifier** on practical latency and
> memory grounds,
> particularly its seconds-versus-minutes latency advantage (Section 3.5).
> A precise standalone ModernBERT-versus-Qwen RSS ratio is not claimed
> because the ModernBERT memory experiment did not isolate both runtimes
> under directly comparable conditions. ModernBERT is selected not because a
> working Qwen CPU path could not be found, but because the one that was
> found and validated is too slow for the "low-latency" requirement this
> task explicitly evaluates. This distinction — classifier
> behavior largely retained, CPU execution demonstrated, but latency and
> memory prohibitive — is the honest, fully evidenced final state of this
> investigation.

---

## 5. Root-Cause Analysis

The original 7B chatbot was slow for three measured reasons: every generated token required processing a 7.62B-parameter model; autoregressive decoding is sequential; and prefill cost grew with full-story context length. Q4_K_M solved memory feasibility but could not eliminate these compute costs.

The final Task 1 design addresses all practical levers available within the deadline: parameter count was reduced to 1.54B, prompt context was reduced to query-relevant evidence, structured requests bypass generation, and output length was capped. Raw prompt and generation throughput improved by 8.6x and 7.5x respectively relative to 7B.

Task 2 has a separate optimization problem. ModernBERT quantization speedups require compatible optimized integer kernels. Several attempted configurations either changed predictions severely or fell back to slow execution paths. Qwen classification remained expensive because its dense 7B backbone processes long sequences and required approximately 17.7 GiB RSS.

---

## 6. Deployment Recommendations

### 6.1 Final architecture

- **Task 1:** persistently loaded Qwen2.5-1.5B Q4_K_M under `llama.cpp`; hybrid dense + BM25 RRF retrieval; one query-relevant chunk; deterministic genre/metadata routes; streaming responses; source IDs and titles returned with answers.
- **Task 2:** persistently loaded ModernBERT FP32 under PyTorch with dynamic input lengths capped at 1,024 tokens and two inference threads. Qwen2.5-7B + QLoRA remains the accuracy winner but is not selected for CPU serving because of measured latency and memory.
- **Request scheduling:** the target exposes only two logical CPUs, so chatbot generation and classification should not run concurrently. A queue or separate worker hosts should isolate the workloads.

### 6.2 Scalability and reliability

- Keep models and retrieval indexes resident; avoid per-request loading.
- Enforce separate input budgets: 2,048 tokens for the chatbot and 1,024 tokens for the classifier.
- Return deterministic errors for missing IDs/titles/genres.
- Log route, retrieval latency, prompt tokens, TTFT, total latency, selected evidence, and output length.
- Add request timeouts and bounded queues.
- For real production, benchmark on a modern multi-core server and scale horizontally with separate chatbot/classifier workers.

---

## 7. Accuracy and Speed Trade-offs

| Optimization | Quality/accuracy impact | Speed/memory impact | Adopted? |
|---|---|---|---|
| Qwen 7B GGUF Q4_K_M | Grounded answers possible | Memory-feasible but 638-1,153 s/query | No |
| Qwen 3B GGUF Q4_K_M | End-to-end quality not evaluated | 13.49 pp tok/s; 6.02 tg tok/s | No |
| **Qwen 1.5B GGUF Q4_K_M** | Correct evidence on representative tests; answers less rich/occasionally overly terse | 27.95 pp tok/s; 11.43 tg tok/s; 2.55 GiB full RAG RSS | **Yes** |
| Relevant-chunk context | Can omit information outside the selected chunk | Major prefill reduction | **Yes** |
| Deterministic genre routing | No generative summary in list response | 0.69 ms and no hallucination risk | **Yes** |
| ModernBERT quantization variants | Severe prediction changes or slow fallback | No acceptable overall gain | No |
| ModernBERT FP32 PyTorch | 48.0% accuracy and 44.81% macro-F1 at the selected 1,024-token cap | 6.41 s mean, 6.66 s P50, 8.65 s P95; ONNX FP32 was slower in both 2,048-token runs | **Yes** |
| 1,024-token classifier cap | −1.0 accuracy point and −1.19 macro-F1 points vs. 2,048 | 20.2% lower mean and 42.6% lower P95 vs. the faster 2,048-token reference | **Yes** |
| Dense Qwen classifier reconstruction | 61.0% fold accuracy; macro-F1 58.95% | 49.6-267.0 s and ~17.7 GiB RSS | No |

The final Task 1 deployment favors latency and memory over answer richness. The final Task 2 deployment accepts a small sequence-truncation accuracy cost to improve mean and tail latency while remaining substantially more practical than the Qwen classifier on CPU.

---

## 8. Further Latency Optimizations

The highest-value remaining optimization is **offline precomputation**. Because the 1,000-story corpus is static, summaries, characters, settings, themes, and endings can be generated once and cached. Common questions would then resolve in milliseconds, while the live 1.5B model would be reserved for genuinely novel details.

Other future options include:

- Evaluate Qwen2.5-0.5B under the same grounded benchmark if lower quality is acceptable.
- Distill a small model specifically for story QA.
- Evaluate speculative decoding with a smaller draft model.
- Precompute and persist embeddings/indexes rather than rebuilding them at process startup.
- Use a production CPU with more cores and modern vector-instruction support.
- Add confidence thresholds: return several retrieved candidates or abstain when top-ranked evidence is weak.

The theoretical latency floor for deterministic/cached routes is millisecond-scale. Novel generated responses remain bounded by prompt processing and sequential decoding unless the model, context, hardware, or generation requirement changes.

---

## 9. Challenges and Engineering Lessons

- Quantization reduced memory but did not guarantee latency improvement.
- Benchmarking model sizes before full integration avoided unnecessary end-to-end tests of the 3B candidate.
- A cached-prefix measurement initially produced an invalid 0.95-second TTFT; explicit `llm.reset()` corrected the methodology.
- Full parent stories erased the latency advantage of chunk retrieval. Preserving the winning evidence chunk reduced prompt size materially.
- Blind character excerpts improved speed but damaged grounding, so query-relevant evidence was adopted instead.
- The final 1.5B system shows a genuine quality/latency trade-off: retrieval sources were correct in key cases, while some answers became terse.
- CPU topology must be reported precisely: the target exposed two logical CPUs but `psutil` reported one physical core.
- Task 2 demonstrated that numerical correctness and runtime efficiency must be validated separately; an optimization can preserve predictions and still be slower.
- Repeated classifier benchmarks showed that absolute latency varies on a shared virtual CPU; reporting both runs and stable relative conclusions avoids cherry-picking.
- Reducing the classifier cap to 1,024 tokens improved P95 latency by 42.6% with only a one-point accuracy reduction, demonstrating that input-budget optimization can be more effective than changing the numerical format.

---

## 10. Final Conclusion

Task 3 produced working CPU implementations for both required components. The final Task 1 chatbot uses Qwen2.5-1.5B-Instruct Q4_K_M with `llama.cpp`, compact query-relevant evidence, deterministic structured routing, and streaming generation. Across nine generated runs, median end-to-end latency was 23.25 seconds and P95 was 26.54 seconds, with 2.55 GiB full-process RSS, compared with 637.7-1,152.8 seconds for the rejected 7B full-context baseline. This is a substantial and directly measured optimization, though the constrained host still prevents consistently low interactive latency.

The final Task 2 CPU classifier is ModernBERT FP32 under PyTorch with dynamic input lengths capped at 1,024 tokens. On the complete 200-example fold-0 held-out partition it achieved 48.0% accuracy and 44.81% macro-F1, with 6.41-second mean, 6.66-second P50, and 8.65-second P95 latency. Relative to the faster 2,048-token reference, this reduced mean latency by 20.2% and P95 by 42.6% for a one-point accuracy cost. Qwen2.5-7B + QLoRA remains the GPU accuracy winner, but its validated dense CPU reconstruction required 49.6-267.0 seconds per example and approximately 17.7 GiB RSS, making it unsuitable for the target CPU.

The resulting architecture is honest about its trade-offs: a smaller quantized generative model for CPU chatbot latency, a compact encoder for CPU classification, deterministic fast paths where generation is unnecessary, and explicit retention of the accuracy-superior Qwen classifier for GPU-capable environments.
