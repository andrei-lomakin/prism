"""CLI entry: `prism eval run` (stage 2), helpers."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prism", description="Prism CLI")
    sub = parser.add_subparsers(dest="cmd")

    eval_p = sub.add_parser("eval", help="Evaluation commands")
    eval_sub = eval_p.add_subparsers(dest="eval_cmd")
    run_p = eval_sub.add_parser("run", help="Run an eval config (stage 2)")
    run_p.add_argument("--config", required=True)

    sub.add_parser("init-db", help="Create DB schema")

    args = parser.parse_args(argv)
    if args.cmd == "init-db":
        from prism.db.session import init_db

        init_db()
        print("OK: schema ready")
        return 0
    if args.cmd == "eval" and args.eval_cmd == "run":
        print("Eval runner lands in stage 2:", args.config)
        return 1
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
