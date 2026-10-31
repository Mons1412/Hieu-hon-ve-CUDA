
## Structured Output Debugging

Observed Qwen2.5 3B response:
{"status":"ANSWER","source_id":3}

Root cause:
The text field was optional with default="".

Patch:
Made text and source_id required fields in
rag_answer_v043.py.

Next:
Verify actual LLM response contains nonempty text.
Then evaluate technical accuracy and citation quality.

## Latest Structured Output Test

Qwen2.5 3B previously returned a one-word answer:
{"status":"ANSWER","text":"Không","source_id":1}

The answer validator correctly rejected it for being too short.

Updated rag_answer_v043.py:
- Require minimum 40 characters for text.
- Ask for a complete explanation.
- Improve diagnostic reasons for invalid output.
- Increase generation token allowance.

Next:
Evaluate answer quality and retrieval faithfulness.

## Latest Progress — Structured Output Fix

Repository:
https://github.com/Mons1412/Hieu-hon-ve-CUDA

Model: qwen2.5:3b

Fixed:
- Required JSON fields
- Minimum answer length
- Improved generation instructions
- Increased num_predict to 320

Latest result:
STATUS: ANSWER
Citation format: PASS

Remaining issue:
The model sometimes produces technically inaccurate
explanations even with valid citations.

Next step:
Evaluate retrieval, answer accuracy, faithfulness,
and abstention behavior before adding more documents.

## RAG Evaluation v0.6

Added evaluate_rag.py.

Evaluation includes:
- Retrieval Hit@4
- Expected answer status
- Out-of-scope question
- JSON result export
- Manual faithfulness review CSV

Known issues:
- Bank Conflict answer incorrectly attributes
  hardware conflict handling to Shared Memory.
- Matrix Multiplication answer sometimes confuses
  uncoalesced memory access with asynchronous access.

Next:
Review evaluation results and improve factual
grounding based on measured failure cases.

## Evaluation Results — 2026-10-31

Commit: 353bd23

Retrieval Hit@4: 2/2
Status Match: 3/3

Bank Conflicts:
ANSWER, partially correct explanation.

Matrix Multiplication:
ANSWER, correct main idea.

Control Flow:
ABSTAIN, requires further review.

Out-of-Scope:
ABSTAIN, correct.

Created EVALUATION_REPORT.md.

Next task:
Expand the evaluation dataset and improve
answer faithfulness before Multi-Document RAG.

## Golden Dataset v0.6.1

Created:
- create_golden_dataset.py
- golden_candidates_v061.csv

Dataset:
20 candidate questions across 5 categories.

Status:
All entries require manual verification.

Retrieval:
Top-4 results can be exported for review.

Next:
Verify reference answers and PDF evidence,
then calculate meaningful evaluation metrics.

## Section-Aware Indexing v0.7.1

Previous issue:
Section heading regex treated bandwidth numbers
such as 144.4, 195.5 and 199.4 as section IDs.

Fix:
Restricted heading recognition and excluded
common bandwidth/frequency units.

New collection:
cuda_pdf_section_v071

Old collections preserved:
cuda_pdf_v03
cuda_pdf_section_v07

Next:
Review section metadata and compare retrieval
on the same document scope.
