# A11 — C3 Fine-Tuning Status

## Status

C3 fine-tuning implementation is complete.

The C3 condition uses an MLX-based LoRA adapter trained on a quantized Llama 3.1 8B base model. Production retrieval uses the SentenceTransformer MiniLM embedding model.

## Hardware and Software

- Hardware: Apple Silicon M4, 16 GB unified memory
- MLX: 0.32.2
- MLX-LM: 0.31.3
- Base model: `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`
- Fine-tuning method: LoRA applied to a 4-bit quantized MLX base model
- LoRA rank: 16
- LoRA scale: 20.0
- LoRA dropout: 0.05
- Adapted layers: 16
- Optimizer: AdamW
- Training iterations: 300
- Batch size: 1
- Gradient accumulation: 2
- Maximum sequence length: 768
- Seed: 42

The implementation is described as LoRA on a 4-bit quantized MLX model. It is not labelled NF4 QLoRA because NF4 quantisation was not independently verified in the implemented MLX workflow.

## Training Dataset

The C3 dataset contains 1,512 programmatically generated examples derived from the engineering-reviewed admissible search space.

Part-level split:

- Training: PILOT_001 to PILOT_008
- Validation: PILOT_009
- Test: PILOT_010

Counts:

- Training: 1,212 examples
- Validation: 138 examples
- Test: 162 examples
- Total: 1,512 examples

Dataset SHA-256:

`e7870a5729ca1dd6ce7e713a846e48250a2967be6a2ee651d98a1780ff4169ce`

The split is grouped by `part_id` to prevent part leakage between training, validation, and test sets.

The examples represent valid or deliberately invalid candidate modifications generated from the engineering-reviewed search space. They are not labelled as globally optimal designs.

## Smoke Training

A 10-iteration smoke run confirmed:

- model loading
- dataset loading
- LoRA attachment
- finite training and validation loss
- adapter saving
- successful adapter reload
- successful MLX generation
- no out-of-memory failure

Peak memory during smoke training was approximately 5.56 GB.

## Main Training

The final training run completed all 300 iterations.

Trainable parameters:

- 20.972 million
- approximately 0.261% of the 8.03 billion model parameters

Peak observed memory was approximately 5.73 GB.

Adapter checkpoints were saved every 50 iterations.

## Validation-Based Checkpoint Selection

All saved checkpoints were evaluated using the full 138-example validation split with prompt masking consistent with training.

| Checkpoint | Validation Loss | Validation Perplexity |
|---:|---:|---:|
| 50 | 1.798 | 6.039 |
| 100 | 3.528 | 34.063 |
| 150 | 1.876 | 6.526 |
| 200 | 1.645 | 5.179 |
| 250 | 1.527 | 4.602 |
| 300 | **1.433** | **4.190** |

Checkpoint 300 was selected because it achieved the lowest full-validation loss among the saved checkpoints.

The canonical adapter is:

`models/c3_adapter/adapters.safetensors`

Its SHA-256 matches the iteration-300 checkpoint:

`8195bde354264b9525764efa995592add09590f8341a2ff2aa5799c7b601d49b`

## Held-Out Test Result

After checkpoint selection was frozen, the untouched PILOT_010 test split was evaluated once.

- Held-out masked test loss: **0.654**
- Held-out perplexity: **1.923**

The held-out result was not used for additional tuning or checkpoint selection.

## Production RAG

Production retrieval was validated with:

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector backend: NumPy cosine index
- Corpus documents: 56
- Corpus chunks: 61
- Retrieval queries: 7
- Recall@5: 1.0000
- Precision@5: 0.2000
- MRR: 1.0000

No development `keyword-hash-256` fallback was used in the production validation.

## C3 Software Integration

The repository includes:

- `MLXLoRABackend`
- lazy model and adapter loading
- explicit adapter validation
- C3 condition routing
- C3 requirement for the MLX LoRA backend
- RAG-enabled C3 path
- default `top_k = 5`
- adapter path and quantisation provenance in generation traces

A direct backend inference test successfully loaded the selected base model and adapter and generated output.

C3 regression tests verify that C3 cannot silently execute through the ordinary non-fine-tuned backend.

## Software Validation

Final repository test result:

`154 passed`

`git diff --check` also completed without errors.

## Scope Limitation

Due to project time constraints, development is frozen at the completed C3 stage.

The C4 autonomous optimisation loop and large-scale final C1-C6 comparative experiment should not be reported as completed unless separately executed and validated.

Existing C5/NSGA-II pilot infrastructure may be retained as development work, but pilot or synthetic results must not be represented as final thesis hypothesis evidence.

## Thesis Interpretation

The implemented evidence supports claims that:

1. an engineering-reviewed material/process search space was constructed;
2. a structured C3 fine-tuning dataset was generated with part-level separation;
3. an MLX-compatible Llama 3.1 8B model was successfully fine-tuned using LoRA;
4. checkpoint selection was performed using validation data;
5. held-out language-model loss was measured only after checkpoint selection;
6. production MiniLM RAG was operational; and
7. the fine-tuned adapter was integrated into the software backend.

It does not by itself establish statistically significant superiority of C3 over C2, Pareto superiority over NSGA-II, or final H1-H4 acceptance.
