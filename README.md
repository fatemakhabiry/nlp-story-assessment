# Story Dataset AI System

An end-to-end NLP system containing a retrieval-augmented story chatbot, a
genre classifier, and CPU-oriented deployment experiments. The project uses
the [`FareedKhan/1k_stories_100_genre`](https://huggingface.co/datasets/FareedKhan/1k_stories_100_genre)
dataset, which contains 1,000 stories across 99 represented genres.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── docs/
│   ├── Task1_Documentation.md
│   ├── Task2_Documentation.md
│   └── Task3_Documentation.md
└── notebooks/
    ├── task1_task2_final.ipynb
    ├── Task3_chatbot_deployment.ipynb
    └── Task3_Classifier_deployment.ipynb
```

## Results summary

| Component | Selected result |
|---|---|
| **Task 1 — Retrieval** | Chunk-level dense retrieval combined with BM25 through reciprocal-rank fusion achieved **67.6% Recall@1**, **88.2% Recall@3**, **94.1% Recall@5**, and **0.795 MRR**. The original whole-story dense baseline achieved 41.2% Recall@1 and 61.8% Recall@3. |
| **Task 2 — Classification** | Qwen2.5-7B + QLoRA achieved **63.8% ± 0.8% mean accuracy**, compared with **52.9% ± 3.2%** for ModernBERT. Qwen was the accuracy winner. |
| **Task 3 — CPU chatbot** | Qwen2.5-1.5B-Instruct in GGUF Q4_K_M format achieved **23.25 s median end-to-end latency** and **21.31 s median time to first token** across nine generated-query runs on the two-core CPU test environment. |
| **Task 3 — CPU classifier** | ModernBERT FP32 with a 1,024-token cap achieved **6.41 s mean latency**, **6.66 s P50**, and **8.65 s P95** on 200 held-out examples, with **48.0% accuracy** and **44.8% macro-F1**. |


Qwen remains the highest-accuracy Task 2 model. ModernBERT is the selected
Task 3 CPU deployment model because it has a substantially smaller resource
footprint and lower inference latency. This is an explicit accuracy–deployment
trade-off; it is not a claim that ModernBERT outperformed Qwen in classification.

Detailed methodology, ablations, limitations, and error analysis are provided
in the corresponding files under `docs/`.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/fatemakhabiry/nlp-story-assessment.git
cd nlp-story-assessment
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Dataset

The dataset is loaded automatically from the Hugging Face Hub. No manual
dataset download is required.

## Runtime requirements

The experiments were developed and executed using Google Colab Pro. GPU notebooks used CUDA-enabled Colab runtimes, while the reported CPU deployment benchmarks used CPU-only Colab runtimes. Colab Pro is not a strict dependency; equivalent environments with compatible hardware and package versions can be used. Because Colab hardware may vary between sessions, latency results should be compared only under the documented CPU and thread conditions.

| Notebook | Recommended runtime | Purpose |
|---|---|---|
| `task1_task2_final.ipynb` | CUDA-capable GPU | Task 1 development, Task 2 baselines, ModernBERT training, Qwen QLoRA training, preservation checks, and the Qwen CPU-feasibility appendix |
| `Task3_chatbot_deployment.ipynb` | CPU-only | GGUF chatbot deployment and end-to-end latency benchmarking |
| `Task3_Classifier_deployment.ipynb` | CPU-only | ModernBERT CPU inference, runtime comparisons, quantization experiments, and sequence-length optimization |

CPU latency results should be reproduced on a CPU-only runtime with the same
thread settings. Merely leaving the GPU unused on a different host does not
make two hardware measurements directly comparable.

## How to run

The notebooks document an iterative experimental workflow containing several
independent and long-running branches. They are not designed for one literal
“Restart and Run All” execution.

1. Open `task1_task2_final.ipynb` in a GPU runtime.
2. Run the common setup and dataset cells first.
3. Run the Task 1 section in order to build and evaluate the final hybrid
   retrieval chatbot.
4. Run the desired Task 2 experimental branch. ModernBERT and Qwen training
   branches may require model cleanup or a fresh runtime because both cannot
   remain in limited GPU memory simultaneously.
5. The Qwen CPU-feasibility appendix records the rejected direct 4-bit merge,
   the fresh dense merge validation, and the tiered CPU timing experiment.
6. Run each Task 3 notebook separately in a fresh CPU-only runtime.

The complete Qwen five-fold cross-validation and other training experiments
take several hours. Their saved outputs and resulting metrics are retained in
the notebooks and documentation; they do not need to be rerun merely to inspect
the submission.

## Task 1 — Retrieval-augmented chatbot

The chatbot uses hierarchical routing:

1. Exact ID lookup for explicit story IDs.
2. Exact title lookup with collision-safe handling.
3. Exact genre lookup for genre-specific requests.
4. Hybrid semantic retrieval for open-ended queries.

The hybrid path divides stories into overlapping chunks, retrieves candidates
with both BGE dense embeddings and BM25, combines rankings with reciprocal-rank
fusion, and maps chunk results back to their parent stories. Retrieved content
is passed to Qwen2.5-7B-Instruct with grounding instructions.

The CPU deployment replaces the 7B generator with Qwen2.5-1.5B-Instruct in
GGUF Q4_K_M format while retaining the hybrid retrieval strategy. The smaller
model was selected after comparing 7B, 3B, and 1.5B CPU feasibility.

## Task 2 — Genre classification

The notebook evaluates:

- TF-IDF with a linear SVM baseline.
- Frozen BGE embeddings with logistic regression.
- ModernBERT-base sequence classification.
- Qwen2.5-7B with QLoRA and masked mean pooling.


For the shared Qwen experiment, LoRA adapters and a classification head are
trained while the original base weights remain frozen. Generation with the
adapter disabled is compared against a deterministic pre-training snapshot to
check that the Task 1 generation behavior is preserved.

## Task 3 — CPU deployment

### Chatbot

- Generator: Qwen2.5-1.5B-Instruct
- Runtime format: GGUF
- Quantization: Q4_K_M
- CPU threads: 2
- Retrieval: chunk-level dense + BM25 + RRF
- Benchmark: three query categories, three runs per category

Across nine runs, retrieval P50 was 53.8 ms, TTFT P50 was 21.31 s, and
end-to-end P50 was 23.25 s.

### Classifier

ModernBERT FP32 was retained after several quantization and runtime experiments
failed to provide a reliable accuracy-preserving speed improvement. Reducing
the input cap from 2,048 to 1,024 tokens produced a **1.25× mean speedup** with
an accuracy reduction of one percentage point on the matched held-out fold.

The fresh dense Qwen merge achieved 5/5 preliminary top-1 agreement with the
original QLoRA classifier. On the complete 200-example fold it achieved 61.0%
accuracy and 58.95% macro-F1. The matching `fold_0_fixed` unmerged checkpoint
achieved 59.5% accuracy and 55.8% macro-F1, so the dense reconstruction changed
these metrics by +1.5 and +3.15 percentage points, respectively. This measured
improvement should be treated as run-specific numerical variation rather than
evidence that merging inherently improves classification. CPU feasibility
testing then required 49.6 s for a 265-token story and 267.0 s for a 1,231-token
story; the near-2,048-token tier was skipped by the predefined hard-stop rule.
These are two feasibility measurements, not a latency distribution.

## Model artifacts

Trained checkpoints and GGUF files are not committed because they range from
tens of megabytes to several gigabytes. The notebooks contain the training and
conversion configurations required to reproduce them. Full reproduction
requires suitable CPU/GPU resources and several hours for cross-validation.

Model artifacts are available upon request.

## Documentation

- [Task 1 Documentation](docs/Task1_Documentation.md): architecture, retrieval evaluation, grounding behavior, and functional tests.
- [Task 2 Documentation](docs/Task2_Documentation.md): preprocessing, baselines, training configurations, cross-validation, statistical comparison, and error analysis.
- [Task 3 Documentation](docs/Task3_Documentation.md): CPU optimization experiments, benchmark methodology, resource use, trade-offs, and production recommendations.

## Production considerations

- Load models and retrieval indexes once during service startup.
- Keep the chatbot and classifier as separate services so they can scale independently.
- Bound input and output lengths and validate incoming requests.
- Stream chatbot tokens to improve perceived responsiveness.
- Cache repeated deterministic lookups and frequently requested stories.
- Record latency, memory, errors, retrieval quality, and prediction drift.
- Use health checks, timeouts, request queues, and controlled concurrency.
- Store model, tokenizer, label-map, and index versions together.

## License and attribution

- Dataset: [`FareedKhan/1k_stories_100_genre`](https://huggingface.co/datasets/FareedKhan/1k_stories_100_genre), used without modifying the underlying records.
- Models: `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `answerdotai/ModernBERT-base`, and `BAAI/bge-small-en-v1.5`.

The dataset and model artifacts remain subject to their respective licenses.
