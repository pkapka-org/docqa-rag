"""Command line: docqa add FILE..., docqa ask "QUESTION", docqa list, docqa remove NAME."""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from .config import Settings
from .llm import LLMError, get_llm
from .loaders import LoadError
from .pipeline import DocumentQA
from .store import DocumentStore, get_embeddings


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="docqa", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("add", help="index files").add_argument("files", nargs="+")
    commands.add_parser("ask", help="ask a question").add_argument("question")
    commands.add_parser("list", help="show indexed files")
    commands.add_parser("remove", help="forget a file").add_argument("name")
    args = parser.parse_args(argv)

    settings = Settings()
    store = DocumentStore(get_embeddings(settings.embedding_model), path=settings.db_path)

    if args.command == "list":
        for name, count in store.sources().items():
            print(f"{name}  ({count} chunks)")
        return 0
    if args.command == "remove":
        store.remove(args.name)
        return 0

    try:
        if args.command == "add":
            qa = DocumentQA(store, None, settings)
            for path in args.files:
                count = qa.add_file(path)
                print(f"{path}: {f'{count} chunks' if count else 'already indexed'}")
            return 0
        answer = DocumentQA(store, get_llm(settings), settings).ask(args.question)
    except (LoadError, LLMError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(answer.text)
    for c in answer.citations:
        print(f"  [{c.number}] {c.source}, {c.location}")
    if not answer.found:
        print(f"  ({answer.reason})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
