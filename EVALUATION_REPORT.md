# CUDA Knowledge Assistant — RAG Evaluation Report

## Project

Repository: Hieu-hon-ve-CUDA

Architecture:
- Ollama + Qwen2.5 3B
- EmbeddingGemma 300M
- ChromaDB
- FastAPI
- Web Chat Interface

## Dataset

Document: NVIDIA CUDA C++ Best Practices Guide

Indexed PDF chunks: 177

Evaluation questions: 4

## Evaluation Results

| Metric | Result |
|---|---|
| Retrieval Hit@4 | 2/2 |
| Expected Status Match | 3/3 |
| Fully Supported Technical Answers | 1/2 preliminary |
| Partially Supported Technical Answers | 1/2 preliminary |
| Out-of-Scope Abstention | 1/1 |

These are small-sample results, not production
accuracy estimates.

## Test Cases

### 1. Bank Conflicts

Status: ANSWER

Retrieval: PASS

PDF pages: 49, 54, 50, 53

Preliminary answer review: PARTIAL

Issue:
The model incorrectly attributes bank-conflict
request splitting to Shared Memory instead of
GPU hardware.

Correct principle:
Shared Memory does not automatically eliminate
bank conflicts. GPU hardware handles conflicting
memory requests, potentially reducing effective
bandwidth.

### 2. Matrix Multiplication

Status: ANSWER

Retrieval: PASS

PDF pages: 52, 50, 53, 51

Preliminary answer review: SUPPORTED

The answer correctly identifies reduced global
memory reads.

Recommended improvement:
Explain tile loading and data reuse between threads.

### 3. Control Flow

Status: ABSTAIN

Review: INCONCLUSIVE

The indexed PDF content and retrieved pages must
be examined before treating abstention as correct.

### 4. Out-of-Scope Question

Question: Thủ đô của Pháp là gì?

Status: ABSTAIN

Review: CORRECT

The CUDA documentation does not support an
answer to this geography question.

## Limitations

- Only four evaluation questions.
- Only two cases have reference PDF pages.
- Page matching does not measure semantic relevance.
- Citation format does not prove factual accuracy.
- Faithfulness judgments are preliminary manual reviews.
- The PDF was indexed only partially.

## Next Steps

1. Expand evaluation dataset to 15-20 questions.
2. Add manually verified expected answers.
3. Review citations against supporting PDF passages.
4. Measure unsupported-claim frequency.
5. Improve generation based on measured errors.
6. Re-run evaluation before Multi-Document RAG.
