#!/usr/bin/env python3
"""Offline regression check for BC01's actor-attribution failure.

This is deliberately a NARROW diagnostic, not a general semantic-faithfulness
classifier. It reads an existing v0.9.2 result; it does NOT call Ollama or edit
production code. All labels remain provisional until human review.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DATA_DIR = Path('eval_results')


def normalized(text: str) -> str:
    return ' '.join(text.casefold().split())


def guard(case: str, answer: str, sources: list[dict]):
    if case != 'BC01/BASELINE':
        return 'NOT_APPLICABLE', 'This rule only covers BC01/BASELINE.'

    relevant_sources = [s.get('text', '') for s in sources]
    source_has_hardware = any(
        re.search(r'the\s+hardware\s+splits\s+a\s+memory\s+request',
                  normalized(s), re.I)
        for s in relevant_sources
    )
    if not source_has_hardware:
        return ('SOURCE_CHECK_REQUIRED',
                'Expected hardware-splits evidence was not found in saved chunks.')

    a = normalized(answer)
    # The observed failure is a claim with a pronoun whose antecedent is
    # Shared Memory. Only flag this specific construction to minimize false positives.
    ambiguous = bool(re.search(
        r'(?:shared memory|bộ nhớ chia sẻ)[^.!?]*[.!?]\s*'
        r'(?:thay vào đó,?\s*)?nó\s+(?:sẽ\s+)?(?:tách|chia)', a
    ))
    explicit_wrong_actor = bool(re.search(
        r'(?:shared memory|bộ nhớ chia sẻ)\s+(?:sẽ\s+)?(?:tách|chia)'
        r'.{0,75}(?:yêu cầu|request)', a
    ))
    if ambiguous or explicit_wrong_actor:
        return ('FAIL_ACTOR_ATTRIBUTION',
                'Answer attributes request-splitting to Shared Memory/pronoun; '
                'source explicitly attributes it to hardware.')

    if ('phần cứng' in a or 'hardware' in a) and re.search(
            r'(?:tách|chia).{0,100}(?:yêu cầu|request)', a):
        return ('ACTOR_CHECK_PASS_ONLY',
                'Hardware actor is explicit; other claims are NOT checked.')

    return ('NEEDS_HUMAN_REVIEW',
            'No matched attribution error; this does NOT prove faithfulness.')


def selftest():
    source = [{'text': 'The hardware splits a memory request that has bank conflicts.'}]
    bad = ('Shared Memory không tự động loại bỏ Bank Conflicts. '
           'Thay vào đó, nó sẽ tách một yêu cầu bộ nhớ có Bank Conflicts.')
    good = ('Không. Khi xảy ra Bank Conflicts, phần cứng GPU sẽ tách '
            'một yêu cầu bộ nhớ thành các yêu cầu không xung đột.')
    assert guard('BC01/BASELINE', bad, source)[0] == 'FAIL_ACTOR_ATTRIBUTION'
    assert guard('BC01/BASELINE', good, source)[0] == 'ACTOR_CHECK_PASS_ONLY'
    assert guard('MM01/BASELINE', bad, source)[0] == 'NOT_APPLICABLE'
    print('OFFLINE SELFTEST: PASS (bad actor / explicit hardware / scope)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selftest', action='store_true')
    parser.add_argument('--input', help='Path to one v0.9.2 JSON result')
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if args.input:
        path = Path(args.input)
    else:
        matches = sorted(DATA_DIR.glob('faithfulness_simple_v092_*.json'))
        if not matches:
            parser.error('No v0.9.2 JSON found. Supply --input PATH.')
        path = matches[-1]
    record = json.loads(path.read_text(encoding='utf-8'))
    case = record.get('case', '')
    answer = record.get('new_answer', '') or ''
    sources = record.get('sources') or []
    outcome, reason = guard(case, answer, sources)
    print('INPUT:', path)
    print('CASE:', case)
    print('MODEL STATUS:', record.get('new_status'))
    print('ACTOR GUARD:', outcome)
    print('REASON:', reason)
    print('NOTE: This is a narrow regression test, not a Faithfulness score.')


if __name__ == '__main__':
    main()
