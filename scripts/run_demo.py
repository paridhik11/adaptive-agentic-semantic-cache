"""Launch live general chat, or select an offline documentation demo."""

import argparse
from contextlib import nullcontext
from getpass import GetPassWarning, getpass
import os
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.api.demo import create_demo_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--directory", type=Path, default=Path("runs/workbench"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--live", action="store_true", help="Free-provider documentation workflow.")
    modes.add_argument("--live-chat", action="store_true", help="General chat with a live free model (the default).")
    modes.add_argument("--offline", action="store_true", help="Replay the local Python documentation demo.")
    modes.add_argument("--cli", action="store_true", help="Run the seven-scenario verification demo.")
    modes.add_argument("--interview", action="store_true", help="Replay the documentation workbench with exact whole-answer reuse.")
    modes.add_argument("--resolve-interrupted", type=int, metavar="ORDINAL",
                       help="After stopping all servers, mark an interrupted live chat attempt as spent.")
    parser.add_argument("--model", help="An explicit OpenRouter :free model ID for live chat.")
    parser.add_argument("--whole-answer-cache", action="store_true", help="Offline exact reuse of approved answers.")
    parser.add_argument("--task-memory", action="store_true", help="Offline scoped local task retrieval.")
    parser.add_argument("--journal", type=Path, help="Existing shared journal, required in live mode.")
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--guarded-rules", action="store_true", help="Opt in to lightweight paraphrase matching in live chat.")
    reuse.add_argument("--experimental-semantic", action="store_true", help="Opt in to guarded semantic reuse.")
    parser.add_argument("--fresh", action="store_true", help="Use temporary storage for this offline web session.")
    args = parser.parse_args()
    if args.resolve_interrupted is not None:
        from src.api.live_chat import DEFAULT_JOURNAL, open_chat_journal

        path = args.journal or DEFAULT_JOURNAL
        if args.resolve_interrupted < 0 or not path.is_file():
            parser.error("Supply a nonnegative attempt ordinal and the existing live chat journal.")
        try:
            resolved = open_chat_journal(path).resolve_interrupted(args.resolve_interrupted)
        except (sqlite3.Error, ValueError):
            parser.error("Could not open the live chat journal. Check its path and file permissions.")
        if not resolved:
            parser.error("No matching reserved attempt at least two minutes old.")
        print(f"Attempt {args.resolve_interrupted} marked interrupted. It still counts against the quota.")
        return
    if args.interview and args.experimental_semantic:
        parser.error("--interview selects the offline exact web demo")
    if args.fresh and (args.live or args.live_chat or args.cli):
        parser.error("--fresh applies to the offline web demo")
    offline_options = args.fresh or args.whole_answer_cache or args.task_memory
    live_chat = args.live_chat or not (args.live or args.offline or args.interview or args.cli or offline_options)
    if args.guarded_rules and not live_chat:
        parser.error("--guarded-rules applies to live chat")
    if args.model and not live_chat:
        parser.error("--model applies to live chat")
    if args.live_chat and (args.whole_answer_cache or args.task_memory):
        parser.error("--live-chat has its own exact cache and follow-up context")
    if args.cli:
        from scripts.demo_flagship import run_demo
        run_demo(args.directory / "flagship_demo", experimental_semantic=args.experimental_semantic)
        return
    if not 1024 <= args.port <= 65535:
        parser.error("Port must be between 1024 and 65535")
    if args.live and not args.journal:
        parser.error("Live mode requires your existing --journal path")

    key = os.environ.get("OPENROUTER_API_KEY", "").strip() if args.live or live_chat else None
    whole_answer = args.interview or args.whole_answer_cache
    task_memory = args.interview or args.task_memory
    profile = "live" if args.live else "reference-whole-answer" if whole_answer else "reference"
    if task_memory:
        profile += "-memory"
    benchmark_directory = ROOT / "dashboard" / "interview"
    if not (benchmark_directory / "report.json").is_file():
        benchmark_directory = None

    import uvicorn

    if live_chat:
        from src.api.live_chat import DEFAULT_JOURNAL, DEFAULT_MODEL, create_live_chat_app

        if not key and sys.stdin.isatty():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", GetPassWarning)
                    key = getpass("OpenRouter API key (input hidden): ").strip()
            except (EOFError, GetPassWarning):
                pass
        if not key:
            parser.error("Set OPENROUTER_API_KEY to launch general chat. Use --interview --fresh for the offline documentation demo.")
        semantic_models = None
        if args.experimental_semantic:
            from src.api.chat_semantic import load_chat_models

            try:
                semantic_models = load_chat_models()
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                parser.error(f"Local semantic models are unavailable. Run the setup in docs/LIVE_CHAT.md. ({type(exc).__name__})")
        try:
            app = create_live_chat_app(
                args.directory / "live-chat", api_key=key, model=args.model or DEFAULT_MODEL,
                journal_path=args.journal or DEFAULT_JOURNAL,
                benchmark_directory=benchmark_directory, semantic_models=semantic_models,
                semantic_mode="guarded" if args.guarded_rules or args.experimental_semantic else "exact",
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(f"LIVE CHAT · {args.model or DEFAULT_MODEL} · http://127.0.0.1:{args.port}", flush=True)
        uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)
        return

    with (TemporaryDirectory(prefix="cache-interview-", ignore_cleanup_errors=True)
          if args.fresh else nullcontext(args.directory)) as directory:
        try:
            app = create_demo_app(
                Path(directory) / profile, live=args.live, journal_path=args.journal, api_key=key,
                whole_answer_cache=whole_answer, task_memory=task_memory,
                experimental_semantic=args.experimental_semantic, benchmark_directory=benchmark_directory,
            )
        except ValueError as exc:
            parser.error(str(exc))

        print(f"Open http://127.0.0.1:{args.port}", flush=True)
        if args.live:
            print("LIVE FREE MODEL — shared journal, fixed 12-attempt allowance.", flush=True)
        else:
            print("REFERENCE REPLAY — scripted answers; real tools, cache and approval; no external model calls.", flush=True)
        if args.fresh:
            print("Fresh demo state is isolated for this session.", flush=True)
        uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
