#!/usr/bin/env python3
"""Optional, narrow Bank Conflicts actor-attribution gate for CUDA RAG API.

Run `python api_gate_v095.py --selftest`, then `--install-api`.
Set CUDA_RESPONSE_GATE=1 to enable on /api/v1/chat; it is OFF by default.
Not a general faithfulness detector, and PASS does not certify an answer.
"""
from __future__ import annotations
import argparse
import ast
import os
import re
import shutil
from pathlib import Path

SAFETY_MSG = (
    "Câu trả lời dự thảo chưa vượt qua bước kiểm tra nguồn. "
    "Vui lòng xem bằng chứng hoặc thử lại."
)


def _norm(value: str) -> str:
    return " ".join((value or "").casefold().split())


def check(question: str, answer: str, sources: list[dict]) -> tuple[str, str]:
    """Rule limited to queries about bank conflicts and attribution of request splits."""
    q = _norm(question)
    a = _norm(answer)
    topic = bool(re.search(r"bank\s+conflicts?|xung\s+đột\s+(?:bank|ngân\s+hàng)", q))
    if not topic:
        return "NOT_APPLICABLE", "Question is outside narrow Bank Conflicts rule."

    hardware_evidence = any(
        re.search(r"the\s+hardware\s+splits\s+a\s+memory\s+request", _norm(src.get("text", "")))
        for src in sources
    )
    if not hardware_evidence:
        return "SOURCE_CHECK_REQUIRED", "Expected source statement about hardware splitting request is missing."

    bad_pronoun = bool(re.search(
        r"(?:shared memory|bộ nhớ chia sẻ)[^.!?]*[.!?]\s*(?:thay vào đó,?\s*)?"
        r"nó\s+(?:sẽ\s+)?(?:tách|chia)", a
    ))
    bad_explicit = bool(re.search(
        r"(?:shared memory|bộ nhớ chia sẻ)\s+(?:sẽ\s+)?(?:tách|chia)"
        r".{0,85}(?:yêu cầu|request)", a
    ))
    if bad_pronoun or bad_explicit:
        return "FAIL_ACTOR_ATTRIBUTION", "Shared Memory incorrectly presented as actor splitting memory requests."

    has_request = bool(re.search(r"yêu\s+cầu|memory\s+request|request", a))
    has_actor = bool(re.search(r"phần\s+cứng|hardware", a))
    pronoun_subject = bool(re.search(r"(?:thay vào đó,?\s*)?nó\s+(?:chỉ\s+)?(?:giảm|tách|chia)", a))
    if has_request and not has_actor and (pronoun_subject or "shared memory" in a):
        return "NEEDS_HUMAN_REVIEW", "Request processing explanation lacks a clear hardware actor."

    return "NO_KNOWN_ACTOR_ERROR", "Narrow check did not find known actor-attribution mistake."


def apply_to_api(question: str, payload: dict) -> dict:
    """Return a new payload; never modify input in-place."""
    result = dict(payload)
    if result.get("status") != "ANSWER":
        result["gate_decision"] = "UNCHANGED"
        return result
    verdict, reason = check(question, str(result.get("answer") or ""), result.get("sources") or [])
    if verdict in {"FAIL_ACTOR_ATTRIBUTION", "SOURCE_CHECK_REQUIRED", "NEEDS_HUMAN_REVIEW"}:
        result["status"] = "INVALID"
        result["answer"] = SAFETY_MSG
        result["reason"] = f"Response Gate: {verdict} ({reason})"
        result["format_validated"] = False
        result["fact_verified"] = False
        result["gate_decision"] = "BLOCK_FOR_REVIEW"
    else:
        result["gate_decision"] = "PASS_THROUGH_UNVERIFIED"
    result["semantic_guard_result"] = verdict
    return result


def selftest() -> None:
    sources = [{"rank": 1, "text": "The hardware splits a memory request that has bank conflicts into separate conflict-free requests."}]
    p = {"status":"ANSWER", "answer": "", "sources":sources, "format_validated":True, "fact_verified":False}
    q = "Shared Memory có tự động loại bỏ Bank Conflicts không?"
    bad = "Shared Memory không tự động loại bỏ Bank Conflicts. Thay vào đó, nó sẽ tách một yêu cầu bộ nhớ. [1]"
    result = apply_to_api(q, dict(p, answer=bad))
    assert result["status"] == "INVALID" and result["gate_decision"] == "BLOCK_FOR_REVIEW"
    assert bad != result["answer"] and "nó sẽ tách" not in result["answer"]
    vague = "Không. Nó chỉ giảm hiệu năng bằng số lượng yêu cầu không có xung đột. [1]"
    result = apply_to_api(q, dict(p, answer=vague))
    assert result["gate_decision"] == "BLOCK_FOR_REVIEW"
    good = "Không. Khi xảy ra Bank Conflicts, phần cứng GPU chia yêu cầu thành các yêu cầu không xung đột. [1]"
    assert apply_to_api(q, dict(p, answer=good))["gate_decision"] == "PASS_THROUGH_UNVERIFIED"
    assert apply_to_api("Matrix Multiplication là gì?", dict(p, answer=bad))["gate_decision"] == "PASS_THROUGH_UNVERIFIED"
    assert apply_to_api(q, dict(p, status="ABSTAIN"))["gate_decision"] == "UNCHANGED"
    assert apply_to_api(q, dict(p, answer=good, sources=[]))["gate_decision"] == "BLOCK_FOR_REVIEW"
    print("OFFLINE SELFTEST: PASS (bad actor / vague actor / good actor / scope / abstain / missing evidence)")


# Exact anchors in api.py checked against public repository main. No changes to its
# generator; patch keeps the existing API behavior with the flag disabled.
ANCHOR_IMPORT = 'from rag_answer_v043 import generate_answer\n'
INJECT_IMPORT = 'from rag_answer_v043 import generate_answer\nfrom api_gate_v095 import apply_to_api\n'
ANCHOR_RESPONSE = '''    return {
        "question": question,
        "status": state,
        "answer": answer,
        "reason": reason,
        "sources": sources,
        "format_validated": state == "ANSWER",
        "fact_verified": False,
    }
'''
INJECT_RESPONSE = '''    result = {
        "question": question,
        "status": state,
        "answer": answer,
        "reason": reason,
        "sources": sources,
        "format_validated": state == "ANSWER",
        "fact_verified": False,
    }
    if os.getenv("CUDA_RESPONSE_GATE", "0") == "1":
        return apply_to_api(question, result)
    return result
'''


def patch_text(original: str) -> str:
    if "from api_gate_v095 import apply_to_api" in original:
        raise ValueError("API gate already installed; no changes made")
    if original.count(ANCHOR_IMPORT) != 1 or original.count(ANCHOR_RESPONSE) != 1:
        raise ValueError("API structure differs from expected version; no changes made")
    modified = original.replace(ANCHOR_IMPORT, INJECT_IMPORT)
    modified = modified.replace(ANCHOR_RESPONSE, INJECT_RESPONSE)
    if "import os\n" not in modified:
        modified = modified.replace("from collections import defaultdict\n", "import os\nfrom collections import defaultdict\n", 1)
    ast.parse(modified)
    return modified


def install() -> None:
    path = Path("api.py")
    if not path.exists():
        raise SystemExit("ERROR: run from the project root containing api.py")
    original = path.read_text(encoding="utf-8")
    try:
        modified = patch_text(original)
    except ValueError as exc:
        raise SystemExit(f"NO CHANGES: {exc}") from exc
    backup = Path("eval_results/api.py.pre_gate_v095.bak")
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        raise SystemExit(f"NO CHANGES: backup {backup} already exists; inspect it first")
    shutil.copy2(path, backup)
    path.write_text(modified, encoding="utf-8")
    print("API PATCH: PASS")
    print("BACKUP:", backup)
    print("DEFAULT: gate OFF; set CUDA_RESPONSE_GATE=1 to enable")
    print("IMPORTANT: test the API locally before deploying")


def smoke_api() -> None:
    """Exercise chat_endpoint with mocked retrieval/LLM: no Ollama, no ChromaDB."""
    from unittest.mock import patch
    import api

    question = "Shared Memory có tự động loại bỏ Bank Conflicts không?"
    sources = [{"rank": 1, "page": 49,
                "text": "The hardware splits a memory request that has bank conflicts into separate conflict-free requests."}]
    bad = ("Shared Memory không tự động loại bỏ Bank Conflicts. "
           "Thay vào đó, nó sẽ tách một yêu cầu bộ nhớ có Bank Conflicts. [1]")
    request = api.ChatRequest(question=question, top_k=4)
    with patch.object(api, "search_evidence", return_value=sources), \
         patch.object(api, "generate_answer", return_value=("ANSWER", bad, "test")):
        with patch.dict(os.environ, {"CUDA_RESPONSE_GATE": "0"}):
            old = api.chat_endpoint(request)
            assert old["status"] == "ANSWER" and old["answer"] == bad
            assert "gate_decision" not in old
        with patch.dict(os.environ, {"CUDA_RESPONSE_GATE": "1"}):
            new = api.chat_endpoint(request)
            assert new["status"] == "INVALID"
            assert new["gate_decision"] == "BLOCK_FOR_REVIEW"
            assert new["answer"] == SAFETY_MSG
    print("API MOCK SMOKE: PASS (flag OFF unchanged / flag ON blocks BC01)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selftest", action="store_true")
    mode.add_argument("--install-api", action="store_true")
    mode.add_argument("--smoke-api", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    elif args.install_api:
        install()
    else:
        smoke_api()

if __name__ == "__main__":
    main()
