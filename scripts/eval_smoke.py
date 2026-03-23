import json
from pathlib import Path

from rag import COLLECTION_NAME, ask_question, hybrid_search

EVAL_PATH = Path(__file__).resolve().parents[1] / "eval_questions.jsonl"


def contains_all(text: str, parts: list[str]) -> bool:
    lowered = text.lower()
    return all(part.lower() in lowered for part in parts)


def main():
    cases = [
        json.loads(line)
        for line in EVAL_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    passed = 0

    for i, case in enumerate(cases, start=1):
        question = case["question"]
        answer = ask_question(question)
        hits = hybrid_search(question, COLLECTION_NAME, limit=3)

        top_source = ""
        if hits:
            top_source = hits[0].payload.get("source", "")

        answer_ok = contains_all(answer, case.get("must_contain", []))
        source_ok = top_source == case.get("expected_source", "")
        ok = answer_ok and source_ok

        if ok:
            passed += 1

        print(f"[{'PASS' if ok else 'FAIL'}] {i}. {question}")
        print(f"answer: {answer}")
        print(f"top_source: {top_source}")
        print()

    print(f"RESULT: {passed}/{len(cases)}")


if __name__ == "__main__":
    main()
