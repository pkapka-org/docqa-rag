"""Score the system on a question set.

    python -m evaluation.run --set test2 --provider extractive

dev    26 questions used to choose settings
test   15 questions, held out at first; seen while debugging, so no longer independent
test2  20 questions written later and scored once, after all settings were fixed

retrieval hit  every expected passage is among the retrieved chunks
answer         the required facts appear in the answer (answerable questions)
citation       the answer cites at least one expected passage
refusal        the system says "not found" (questions the documents do not answer)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from docqa.config import Settings
from docqa.llm import PROVIDERS, get_llm
from docqa.pipeline import Citation, DocumentQA
from docqa.store import DocumentStore, get_embeddings

HERE = Path(__file__).resolve().parent
SAMPLES = HERE.parent / "sample_docs"


def matches(citation: Citation, expected: dict) -> bool:
    if citation.source != expected["source"]:
        return False
    if "page" in expected:
        return citation.location == f"page {expected['page']}"
    if "section" in expected:
        return citation.location.endswith(expected["section"])
    return True


def has_facts(text: str, facts: list[list[str]]) -> bool:
    text = text.lower()
    return all(any(option.lower() in text for option in options) for options in facts)


def evaluate(qa: DocumentQA, questions: list[dict]) -> list[dict]:
    rows = []
    for item in questions:
        start = time.perf_counter()
        answer = qa.ask(item["question"])
        row = {"id": item["id"], "type": item["type"], "question": item["question"],
               "answer": answer.text, "found": answer.found, "reason": answer.reason,
               "cited": [f"{c.source}, {c.location}" for c in answer.citations],
               "top_score": round(answer.retrieved[0].score, 3) if answer.retrieved else None,
               "seconds": round(time.perf_counter() - start, 2)}
        if item["type"] == "absent":
            row["refusal"] = not answer.found
        else:
            row["retrieval"] = all(
                any(matches(c, s) for c in answer.retrieved) for s in item["sources"])
            row["answer_ok"] = answer.found and has_facts(answer.text, item["must_include"])
            row["citation"] = any(matches(c, s) for c in answer.citations for s in item["sources"])
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return rows


def rate(rows: list[dict], key: str) -> str:
    values = [r[key] for r in rows if key in r]
    return f"{sum(values)}/{len(values)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=("dev", "test", "test2"), default="test")
    parser.add_argument("--provider", choices=PROVIDERS, default="extractive")
    parser.add_argument("--model", default="")
    parser.add_argument("--save", help="write the rows to this JSON file")
    args = parser.parse_args()

    settings = Settings(provider=args.provider, model=args.model)
    store = DocumentStore(get_embeddings(settings.embedding_model))
    qa = DocumentQA(store, get_llm(settings), settings)
    for path in sorted(SAMPLES.iterdir()):
        qa.add_file(str(path))

    rows = evaluate(qa, json.loads((HERE / f"{args.set}.json").read_text()))
    times = sorted(r["seconds"] for r in rows)
    summary = {"set": args.set, "provider": args.provider, "model": settings.resolved_model,
               "retrieval": rate(rows, "retrieval"), "answer": rate(rows, "answer_ok"),
               "citation": rate(rows, "citation"), "refusal": rate(rows, "refusal"),
               "median_seconds": times[len(times) // 2]}
    print(json.dumps(summary, indent=1))
    if args.save:
        Path(args.save).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
