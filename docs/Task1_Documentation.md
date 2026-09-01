# Task 1: Story Retrieval Chatbot - Updated Documentation

## 1. Executive summary

Task 1 implements a retrieval-augmented chatbot over all 1,000 stories in `FareedKhan/1k_stories_100_genre`. The final architecture uses deterministic routing for structured requests and chunk-level hybrid retrieval for open-ended content questions. Retrieved stories are supplied to `Qwen2.5-7B-Instruct`, loaded with 4-bit NF4 quantization, under an instruction that answers must remain grounded in the supplied text.

**Note on model size across tasks**: Qwen2.5-7B-Instruct is the model used throughout Task 1's own development, evaluation, and retrieval benchmarking (this document), and is the model whose weights are shared with Task 2's QLoRA classifier and validated for catastrophic-forgetting preservation. Task 3's CPU deployment separately conducts a model-size ablation (7B/3B/1.5B) and selects **Qwen2.5-1.5B-Instruct** specifically for the CPU-only production target, given the 7B model's CPU latency proved impractical. This is a deliberate, evidence-based environment-specific decision — see Task 3 documentation, Section 2.3 — not a contradiction of this document's own findings, which remain the valid basis for Task 1/2's GPU-based architecture and accuracy claims.

The initial one-vector-per-story dense retriever was evaluated rather than accepted by assumption. It achieved only 61.8% Recall@3 on a 34-query benchmark. Overlapping chunk retrieval improved Recall@3 to 73.5%. Adding BM25 and Reciprocal Rank Fusion increased Recall@3 to 88.2%, Recall@10 to 97.1%, Recall@20 to 100%, and MRR@20 to 0.795. The hybrid retriever was therefore selected as the final design.

## 2. Dataset validation

The source dataset contains:

- 1,000 rows.
- Four columns: `story`, `genre`, `title`, and `id`.
- 1,000 unique IDs.
- 998 unique case-insensitive titles.
- 99 unique genre labels in the actual data.

The assessment describes 100 genres. Inspection of the supplied `story_genres.pkl` found 100 entries but only 99 unique values because `Historical Adventure` appears twice. No row or label was added, removed, or relabeled; the fixed dataset was used as supplied.

Two titles collide case-insensitively:

- `The Voyage of the Sea Serpent` corresponds to IDs 735471 and 623181.
- `The Chronicles of the Lost Kingdom` corresponds to IDs 320505 and 632633.

The paired stories have different text and are legitimate distinct records. Title lookup therefore maps a normalized title to a list of rows and returns every match.

## 3. Requirements mapping

| Requirement | Implementation |
|---|---|
| Access all 1,000 stories | Exact lookup tables and retrieval indexes cover all rows |
| Query by ID | O(1) dictionary lookup |
| Query by title | Case-insensitive collision-safe lookup |
| Query by genre | Deterministic genre bucket lookup |
| Query by themes/content/details | Chunk-level dense + BM25 hybrid retrieval |
| Accurate story-grounded answers | Retrieved text is supplied to Qwen with explicit grounding and abstention instructions |
| Resource constraint | Qwen2.5-7B loaded in 4-bit NF4; original run used approximately 5.3 GiB allocated VRAM |
| Demonstrate functionality | Retrieval benchmark, latency measurements, core query tests, edge cases, and source metadata |

## 4. Final architecture

```text
User query
    |
    v
Query router
    |-- ID request ------> exact ID lookup
    |-- Title request ---> collision-safe title lookup
    |-- Genre request ---> deterministic metadata listing
    `-- Content request
            |-- dense search over overlapping chunks
            `-- BM25 search over the same chunks
                     |
                     v
             Reciprocal Rank Fusion
                     |
                     v
             Group by parent story ID
                     |
                     v
          Top distinct stories + metadata
                     |
                     v
        Qwen2.5-7B-Instruct (4-bit NF4)
                     |
                     v
             Answer + source IDs/titles
```

Structured requests bypass semantic retrieval because exact lookup is more reliable and cheaper. Title routing occurs before genre parsing to prevent title questions such as `What genre is the story called ...?` from being misrouted as unknown genres.

## 5. Retrieval design

### 5.1 Baseline: complete-story dense embeddings

The baseline concatenated each title and complete story and encoded the result with `BAAI/bge-small-en-v1.5`. Embeddings were L2-normalized and indexed with FAISS `IndexFlatIP`, making inner product equivalent to cosine similarity.

The advantage was simplicity: only 1,000 vectors were required. Its weakness was dilution of localized information. A minor event, name, quotation, or object could have little influence on the representation of a story averaging approximately 1,208 Qwen tokens.

### 5.2 Overlapping chunks

Stories were divided into 500-word chunks with 100-word overlap. This generated 2,764 chunks, averaging 2.8 chunks per story. Each record preserves:

- Story ID.
- Title.
- Genre.
- Chunk index.
- Chunk text.

Dense hits are grouped by story ID using the highest-scoring chunk, preventing several chunks from one story from occupying multiple story-level ranks.

### 5.3 BM25 and hybrid fusion

BM25 complements dense embeddings by emphasizing rare lexical evidence such as character names, project names, spacecraft names, quoted expressions, and unusual objects. This was important for queries containing details such as `Project Genesis`, `SS Excelsior`, `beta-carotene`, `Mr. Black`, and `twisted willow tree`.

Dense and BM25 rankings are fused at chunk level using Reciprocal Rank Fusion:

\[
\operatorname{RRF}(d)=\sum_r \frac{1}{k+\operatorname{rank}_r(d)}
\]

with `k=60`. Fused chunks are then grouped into distinct stories using the maximum fused chunk score.

## 6. Retrieval evaluation

### 6.1 Methodology

The evaluation set contains 34 manually verified semantic queries with known target story IDs. It includes:

- Distinctive named entities and events.
- Paraphrased themes and plots.
- Localized details, dialogue, and minor events.

Every system used the same queries, target IDs, cutoffs, and `MRR@20` implementation. Recall@k counts a query as successful when its expected story appears within the first k distinct story results.

### 6.2 Ablation results

| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Recall@20 | MRR@20 |
|---|---:|---:|---:|---:|---:|---:|
| Complete-story dense | 41.2% | 61.8% | 61.8% | 73.5% | 85.3% | 0.528 |
| Chunk dense | 52.9% | 73.5% | 79.4% | 85.3% | 91.2% | 0.653 |
| **Chunk hybrid: dense + BM25 + RRF** | **67.6%** | **88.2%** | **94.1%** | **97.1%** | **100.0%** | **0.795** |

Relative to the original dense baseline, hybrid retrieval improved both Recall@1 and Recall@3 by 26.4 percentage points. It retrieved the correct story for all benchmark queries within the first 20 results.

### 6.3 Recursive-splitting ablation

A boundary-aware recursive splitter was tested with approximately the same 500-word chunk size and 100-word overlap. Compared with fixed windows, its dense retrieval reduced Recall@1 from 52.9% to 47.1%, Recall@3 from 73.5% to 70.6%, and MRR@20 from 0.653 to 0.615. With hybrid BM25–dense fusion, it matched Recall@1 (67.6%) and Recall@3 (88.2%) and improved Recall@5 from 94.1% to 97.1%, but reduced Recall@20 from 100.0% to 97.1% and MRR@20 from 0.795 to 0.785. Because it provided no consistent overall improvement, the simpler fixed 500-word splitter with 100-word overlap was retained.

### 6.4 Latency

Across the 34 benchmark queries, measured hybrid retrieval latency was:

| Statistic | Latency |
|---|---:|
| Mean | 31.7 ms |
| Median | 31.0 ms |
| P95 | 40.1 ms |
| Minimum | 25.2 ms |
| Maximum | 41.3 ms |

These results measure retrieval on the original test host. They do not represent Qwen generation latency or a universal hardware-independent figure.

### 6.5 Reranker decision

A cross-encoder reranker was considered but not adopted. Hybrid retrieval already achieved 88.2% Recall@3, 97.1% Recall@10, and 100% Recall@20. Another neural stage could potentially improve top-rank ordering, but would add CPU latency, memory consumption, dependencies, and operational complexity. Given the CPU deployment objective, the expected incremental benefit did not justify the additional serving cost.

## 7. LLM selection and generation

### 7.1 Alternatives considered

| Model | Reason not selected |
|---|---|
| Llama-3.1-8B-Instruct | Close competitor at comparable size and quality. Qwen2.5-7B was preferred for instruction-following and long-context handling; the margin was not decisive. |
| Mistral-7B-Instruct-v0.3 | Benchmarks slightly behind Qwen2.5-7B on recent instruction/reasoning leaderboards at similar parameter count. |
| Closed APIs (GPT-4, Claude, etc.) | Rejected outright. The task requires local, self-hosted deployment within a fixed VRAM budget and CPU-only deployment in Task 3; an API-based model cannot satisfy either constraint. |
| Llama-3.1-70B / Qwen2.5-72B | Would likely exceed 7B-class quality, but require roughly 140+ GiB in fp16 — incompatible with the 24 GiB combined VRAM budget even under aggressive quantization. |
| Small instruct models (Phi-3-mini, TinyLlama, ~1-3B) | Trivially fit the VRAM budget, but carry materially higher risk of weaker instruction-following and less detailed, less accurate answers — a direct concern given the brief explicitly grades answer accuracy and relevance. |

Qwen2.5-7B-Instruct was selected as the model offering the best balance of
instruction-following quality, context length, and a VRAM footprint
compatible with sharing the 24 GiB budget against Task 2's classifier.
This selection governs Task 1's own development and evaluation (this
document) and Task 2's shared-instance classifier. Task 3 separately
re-evaluates model size specifically for CPU deployment latency and
selects a smaller variant for that distinct environment — see Task 3
documentation, Section 2.3, for that ablation and its reasoning.

### 7.2 Configuration

`Qwen/Qwen2.5-7B-Instruct` was selected because it provides strong instruction following and summarization, supports long contexts, and is small enough to load in 4-bit form within the project constraint. The configuration uses:

- NF4 4-bit weights.
- Double quantization.
- BF16 computation.
- Hugging Face chat template.
- Deterministic decoding in the cleaned implementation (`do_sample=False`).

The original GPU run reported approximately 5.32 GiB allocated VRAM after loading the model and approximately 6.19 GiB peak allocated memory during representative generation. This remained well below the 24 GB combined project limit.

## 8. Context construction and grounding

For ID and title queries, complete story text is supplied. For semantic queries, hybrid retrieval selects the most relevant distinct stories and those stories are supplied with explicit IDs, titles, and genres. The prompt requires the model to:

- Use only the supplied context.
- Keep multiple stories separate.
- Avoid inventing unsupported details.
- State when the supplied text does not support an answer.

Genre queries use a deterministic metadata list instead of asking the model to infer ten plots from tiny excerpts. Users can select a title or ID for a complete summary.

The chatbot returns structured source metadata with every answer, improving traceability. A dialogue heuristic detects whether quoted speech exists and instructs the model not to fabricate quotations when the source contains only narrated conversations.

## 9. Functional testing

The final system covers:

- Existing and nonexistent IDs.
- Unique and colliding titles.
- Existing and nonexistent genres.
- Broad thematic queries.
- Fine-grained detail queries.
- Dialogue and quotation questions.
- Nonsensical or unsupported queries.

Representative verified examples include:

- ID 523790, `Willowbrook Manor's Ghostly Echoes`.
- Both records titled `The Voyage of the Sea Serpent`.
- Science Fiction genre listing.
- Adventure-and-friendship semantic search.
- Direct-dialogue verification.

## 10. Resource efficiency

The retrieval corpus is small enough for exact FAISS search and in-memory BM25:

- 1,000 story vectors for the baseline.
- 2,764 chunk vectors for the final dense index.
- 2,764 BM25 documents.
- 384 dimensions per BGE embedding.

Retrieval overhead is negligible relative to Qwen generation. The embedding model and indexes are reusable and should remain resident in a long-running service rather than being rebuilt per request.

## 11. Challenges and engineering decisions

### Dataset mismatch

The advertised 100 genres reduce to 99 unique labels because of a duplicated genre entry. The dataset was preserved and the discrepancy documented.

### Duplicate titles

A one-value title dictionary silently dropped valid stories. Collision-safe lists preserve all records and the response disambiguates them by ID.

### Weak detail retrieval

The initial dense baseline missed many localized details. Measurement showed 61.8% Recall@3. Chunking improved localized representation, and BM25 recovered rare lexical evidence. The measured ablation justified both additions.

### Genre-answer speculation

Generating descriptions from only the first 200 characters could encourage unsupported extrapolation. The cleaned design returns deterministic metadata for genre listings and reserves full generation for selected stories.

### Complexity control

A reranker was rejected because its unmeasured incremental improvement did not justify extra CPU cost after hybrid retrieval achieved strong recall.

## 12. Limitations and future work

- The retrieval evaluation has only 34 manually curated queries; a larger independently authored set would provide stronger evidence.
- Recall@3 is 88.2%, so some questions can still omit the correct story from a three-story context.
- A production system should add calibrated confidence thresholds, abstention, request logging, and monitoring for retrieval drift and unsupported answers.
- BM25 uses simple lowercase whitespace tokenization. A more robust analyzer could improve punctuation, possessives, and morphological matching.
- If ranking requirements become stricter and CPU capacity permits, a lightweight reranker could be evaluated as an ablation over the top 10 candidates.

## 13. Final conclusion

The final system combines deterministic structured lookup with measured hybrid semantic retrieval. The design evolved from a simple complete-story dense baseline to chunked dense retrieval and finally to dense + BM25 RRF fusion based on observed failures and ablation results. The selected hybrid system provides materially stronger coverage while preserving low retrieval latency and avoiding the extra operational cost of a reranker. Qwen2.5-7B-Instruct then produces grounded answers with explicit source metadata under the available GPU budget.
