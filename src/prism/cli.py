"""CLI entry: `prism init-db`, `prism ingest-demo`, `prism golden build`, `prism eval run`."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prism", description="Prism CLI")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("init-db", help="Create DB schema + pgvector extension")
    sub.add_parser("rebuild-bm25", help="Rebuild BM25 index from indexed chunks")

    demo_p = sub.add_parser("ingest-demo", help="Index data/demo/corpus into Postgres")
    demo_p.add_argument("--space", default="shared")

    golden_p = sub.add_parser("golden", help="Golden dataset commands")
    golden_sub = golden_p.add_subparsers(dest="golden_cmd")
    build_p = golden_sub.add_parser("build", help="Write data/golden/{dev,test}.jsonl")
    build_p.add_argument("--target", type=int, default=240)

    eval_p = sub.add_parser("eval", help="Evaluation commands")
    eval_sub = eval_p.add_subparsers(dest="eval_cmd")
    run_p = eval_sub.add_parser("run", help="Run an eval config (stage 2)")
    run_p.add_argument("--config", required=True)
    run_p.add_argument(
        "--skip-generation",
        action="store_true",
        help="Force retrieval-only even if config enables generation",
    )
    run_p.add_argument(
        "--heuristic-judge",
        action="store_true",
        help="Skip LLM judge; use lexical heuristics only",
    )

    args = parser.parse_args(argv)

    if args.cmd == "init-db":
        from prism.db.session import init_db

        init_db()
        print("OK: schema ready")
        return 0

    if args.cmd == "rebuild-bm25":
        from prism.db.session import get_session_factory, init_db
        from prism.rag.index.rebuild import rebuild_bm25

        init_db()
        db = get_session_factory()()
        try:
            n = rebuild_bm25(db)
        finally:
            db.close()
        print(json.dumps({"bm25_chunks": n}, ensure_ascii=False))
        return 0

    if args.cmd == "ingest-demo":
        from prism.ingest.demo import ingest_demo_corpus

        summary = ingest_demo_corpus(space=args.space)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary.get("failed_jobs", 0) == 0 else 1

    if args.cmd == "golden" and args.golden_cmd == "build":
        from prism.eval.golden_build import build

        summary = build(target=args.target)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "eval" and args.eval_cmd == "run":
        from prism.eval.runner import run_eval

        summary = run_eval(
            args.config,
            skip_generation=bool(args.skip_generation),
            use_llm_judge=not bool(args.heuristic_judge),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
