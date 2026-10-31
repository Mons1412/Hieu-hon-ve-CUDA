#!/usr/bin/env python3
"""Hieu-hon-ve-CUDA v0.9.4: apply a narrow semantic safety gate to saved answers.

This is an OFFLINE regression experiment; it does not call Ollama, query
ChromaDB, replace the production generator, or claim general faithfulness.

The guard detects the observed BC01/BASELINE mistake: attributing a CUDA
hardware memory-request split to the Shared Memory storage region itself.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from subject_guard_v093 import guard

DATA_DIR = Path("eval_results")


def apply_gate(record: dict) -> dict:
    required = ("case", "new_status", "new_answer", "sources")
    missing = [k for k in required if k not in record]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")
    if not isinstance(record["sources"], list):
        raise ValueError("sources must be a list")

    status = record["new_status"]
    answer = record["new_answer"] or ""
    if status != "ANSWER":
        verdict, reason = "NO_ANSWER_TO_CHECK", "Candidate status is not ANSWER."
        decision, safe_answer = "UNCHANGED", answer
    else:
        verdict, reason = guard(record["case"], answer, record["sources"])
        if verdict in {"FAIL_ACTOR_ATTRIBUTION", "SOURCE_CHECK_REQUIRED", "NEEDS_HUMAN_REVIEW"}:
            decision = "BLOCK_FOR_REVIEW"
            safe_answer = (
                "Câu trả lời dự thảo chưa vượt qua bước kiểm tra nguồn. "
                "Vui lòng xem bằng chứng hoặc thử lại."
            )
        else:
            decision = "PASS_THROUGH_UNVERIFIED"
            safe_answer = answer

    return {
        "case": record["case"],
        "original_model_status": status,
        "original_answer": answer,
        "semantic_guard_result": verdict,
        "semantic_guard_reason": reason,
        "gate_decision": decision,
        "display_answer": safe_answer,
        "source_reference": record.get("evidence_source_json"),
        "warning": (
            "A non-blocking result does NOT prove semantic faithfulness. "
            "This gate only covers a single known BC01 actor-attribution failure."
        ),
    }


def selftest() -> None:
    evidence = [{"rank": 1, "text": (
        "The hardware splits a memory request that has bank conflicts "
        "into separate conflict-free requests."
    )}]
    bad = {
        "case": "BC01/BASELINE", "new_status": "ANSWER", "sources": evidence,
        "new_answer": (
            "Shared Memory không tự động loại bỏ Bank Conflicts. "
            "Thay vào đó, nó sẽ tách một yêu cầu bộ nhớ có Bank Conflicts. [1]"
        ),
    }
    blocked = apply_gate(bad)
    assert blocked["semantic_guard_result"] == "FAIL_ACTOR_ATTRIBUTION"
    assert blocked["gate_decision"] == "BLOCK_FOR_REVIEW"
    assert "nó sẽ tách" not in blocked["display_answer"]

    corrected = dict(bad, new_answer=(
        "Không. Khi có Bank Conflicts, phần cứng GPU chia yêu cầu truy cập "
        "thành các yêu cầu không xung đột. [1]"
    ))
    check = apply_gate(corrected)
    assert check["semantic_guard_result"] == "ACTOR_CHECK_PASS_ONLY"
    assert check["gate_decision"] == "PASS_THROUGH_UNVERIFIED"

    unrelated = dict(bad, case="MM01/BASELINE")
    assert apply_gate(unrelated)["semantic_guard_result"] == "NOT_APPLICABLE"

    non_answer = dict(bad, new_status="ABSTAIN", new_answer="")
    assert apply_gate(non_answer)["gate_decision"] == "UNCHANGED"
    print("OFFLINE SELFTEST: PASS (block bad / accept limited / scope / abstain)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--input", help="Saved faithfulness_simple_v092_*.json")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return

    if args.input:
        source = Path(args.input)
    else:
        candidates = sorted(DATA_DIR.glob("faithfulness_simple_v092_*.json"))
        if not candidates:
            parser.error("No saved v0.9.2 result found. Pass --input PATH.")
        source = candidates[-1]

    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        parser.error("Expected one v0.9.2 JSON record (not an array).")
    outcome = apply_gate(data)
    outcome["input_file"] = str(source)

    DATA_DIR.mkdir(exist_ok=True)
    output = DATA_DIR / (
        "response_gate_v094_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json"
    )
    temp = output.with_suffix(".json.tmp")
    temp.write_text(json.dumps(outcome, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(output)

    print("CASE:", outcome["case"])
    print("GUARD:", outcome["semantic_guard_result"])
    print("DECISION:", outcome["gate_decision"])
    print("ORIGINAL:", outcome["original_answer"])
    print("SAFE DISPLAY:", outcome["display_answer"])
    print("REASON:", outcome["semantic_guard_reason"])
    print("SAVED:", output)
    print("IMPORTANT: A pass-through is NOT a general faithfulness verification.")


if __name__ == "__main__":
    main()
