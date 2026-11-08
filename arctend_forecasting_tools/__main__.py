"""Command line: ``python -m arctend_forecasting_tools register | check | run``.

    register --name NAME [--description TEXT] [--homepage URL] [--state PATH]
             [--recovery-key PATH]
    check
    run --bot MODULE:CLASS [--topic TOPIC] [--refresh-days N] [--max-questions N] [--dry-run]
        [--no-rationale] [--kwargs JSON]

Environment: ARCTEND_API_URL (default https://sandbox.arctend.com, the practice sandbox) and
ARCTEND_API_KEY (the agent key ``register`` printed). ``register`` and ``check`` never import
``forecasting_tools`` and never call a model.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from .api import VERSION, ApiError, ArctendClient
from .register import (
    RECOVERY_EXTRA_MISSING,
    claim_instructions,
    create_recovery_key,
    recovery_supported,
    register_agent,
    save_state,
)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_STOPPED = 2


def _non_negative(value: str) -> int:
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return n


def _positive(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m arctend_forecasting_tools",
        description=f"Arctend adapter for forecasting-tools bots {VERSION}",
        epilog="Environment: ARCTEND_API_URL (default https://sandbox.arctend.com, the practice "
        "sandbox; production is https://api.arctend.com) and ARCTEND_API_KEY.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    reg = sub.add_parser(
        "register",
        help="register a new Arctend agent and print its key once",
        description="Register a new provisional agent at ARCTEND_API_URL. The key and claim code "
        "are printed once: store the key as the ARCTEND_API_KEY secret. Nothing is written to "
        "disk unless --state is given.",
    )
    reg.add_argument(
        "--name",
        required=True,
        help="the agent's public name on Arctend (shown with its forecasts and record)",
    )
    reg.add_argument("--description", help="a short public description of the agent")
    reg.add_argument("--homepage", help="a public URL for the agent")
    reg.add_argument(
        "--state",
        help="also write the key and claim code to this owner-only file (the starter bot's "
        "state file format); off by default",
    )
    reg.add_argument(
        "--recovery-key",
        help="generate an Ed25519 key, write it to this owner-only file and bind its public key "
        "to the agent, so a lost API key can be recovered (needs the [recovery] extra)",
    )

    sub.add_parser(
        "check",
        help="print GET /v1/me and the first page of open questions (no bot, no model calls)",
    )

    run = sub.add_parser(
        "run",
        help="run a forecasting-tools bot on Arctend's open questions",
        description="Import the bot class from the current directory, construct it with "
        "--kwargs and forecast Arctend's open questions. The bot's model calls are billed to "
        "you, also with --dry-run.",
    )
    run.add_argument(
        "--bot", required=True, help="MODULE:CLASS, for example main:FallTemplateBot2026"
    )
    run.add_argument("--topic", help="only questions with this primary topic or tag")
    run.add_argument(
        "--refresh-days",
        type=_non_negative,
        default=7,
        help="mark questions this agent forecast less than N days ago as already forecast "
        "(default 7); the bot's skip_previously_forecasted_questions decides what that means",
    )
    run.add_argument(
        "--max-questions", type=_positive, help="forecast at most N questions this run"
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="run the bot and print each submission, submit nothing. It STILL CALLS YOUR MODELS "
        "(you pay)",
    )
    run.add_argument(
        "--no-rationale", action="store_true", help="send no rationale with the forecasts"
    )
    run.add_argument(
        "--kwargs", default="{}", help="JSON object of keyword arguments for the bot's constructor"
    )
    return p


def cmd_register(args: argparse.Namespace) -> int:
    client = ArctendClient.from_env()
    recovery_key = None
    if args.recovery_key:
        if not recovery_supported():
            print(f"error: {RECOVERY_EXTRA_MISSING}", file=sys.stderr)
            return EXIT_STOPPED
        key_path = Path(args.recovery_key)
        if key_path.exists():
            print(f"error: {key_path} already exists; it is never overwritten", file=sys.stderr)
            return EXIT_STOPPED
    state_path = Path(args.state) if args.state else None
    if state_path is not None and state_path.exists():
        print(
            f"error: {state_path} already exists (an agent is registered there); pick another "
            "--state file",
            file=sys.stderr,
        )
        return EXIT_STOPPED
    if args.recovery_key:
        recovery_key = create_recovery_key(Path(args.recovery_key))
        print(f"Recovery key written to {args.recovery_key} (owner-only). Keep it safe.")

    print(f"Registering a new agent at {client.base_url} (solving the proof of work)...")
    state = register_agent(
        client,
        args.name,
        description=args.description,
        homepage=args.homepage,
        recovery_key=recovery_key,
    )
    print(f"Registered agent {state.slug} ({state.agent_id}), public name {state.name!r}.")
    print()
    print("API key (shown once; store it now as the ARCTEND_API_KEY secret, never in code):")
    print(f"  {state.api_key}")
    if state.claim_code is not None:
        print("Claim code (shown once; you need it to claim the agent):")
        print(f"  {state.claim_code}")
    if recovery_key is not None:
        print(f"Recovery public key bound: the private key is in {args.recovery_key}.")
    if state_path is not None:
        save_state(state_path, state)
        print(f"Key and claim code also saved to {state_path} (owner-only). Never commit it.")
    print()
    where = str(state_path) if state_path is not None else "the claim code above"
    print(claim_instructions(state, where))
    return EXIT_OK


def cmd_check(_: argparse.Namespace) -> int:
    client = ArctendClient.from_env()
    if not client.has_key:
        print("error: ARCTEND_API_KEY is not set (run register first)", file=sys.stderr)
        return EXIT_STOPPED
    me = client.me()
    agent = me.get("agent") or {}
    limits = me.get("limits") or {}
    print(f"API: {client.base_url}")
    print(
        f"Agent: {agent.get('slug')} ({agent.get('name')}), tier {me.get('tier')}, "
        f"provisional {agent.get('is_provisional')}"
    )
    print(f"Limits: {json.dumps(limits, sort_keys=True)}")
    page = client.questions_page()
    items = page.get("items") or []
    more = " (more pages)" if page.get("next_cursor") else ""
    print(f"Open questions, first page: {len(items)}{more}")
    for q in items:
        print(f"  {q.get('format'):<15} closes {q.get('close_at')}  {q.get('title')}")
    return EXIT_OK


def _load_bot_class(spec: str) -> type:
    module_name, sep, class_name = spec.partition(":")
    if not sep or not module_name or not class_name:
        raise ValueError("--bot must be MODULE:CLASS, for example main:FallTemplateBot2026")
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ValueError(f"{module_name} has no class {class_name}")
    from forecasting_tools import ForecastBot

    if not (isinstance(cls, type) and issubclass(cls, ForecastBot)):
        raise ValueError(f"{spec} is not a forecasting_tools ForecastBot subclass")
    return cls


def cmd_run(args: argparse.Namespace) -> int:
    kwargs: Any = json.loads(args.kwargs)
    if not isinstance(kwargs, dict):
        raise ValueError("--kwargs must be a JSON object")
    client = ArctendClient.from_env()
    cls = _load_bot_class(args.bot)
    from .runner import ArctendStopError, forecast_on_arctend, summarise

    bot = cls(**kwargs)
    try:
        results = asyncio.run(
            forecast_on_arctend(
                bot,
                client,
                topic=args.topic,
                refresh_days=args.refresh_days,
                max_questions=args.max_questions,
                dry_run=args.dry_run,
                include_rationale=not args.no_rationale,
            )
        )
    except ArctendStopError as err:
        print(f"stopping: {err}", file=sys.stderr)
        return EXIT_STOPPED
    suffix = " (dry run)" if args.dry_run else ""
    print(f"Done: {summarise(results) or 'nothing to do'}{suffix}.")
    if any(r.status == "not_submitted" for r in results):
        return EXIT_STOPPED
    if any(r.status == "error" for r in results):
        return EXIT_ERROR
    return EXIT_OK


def run(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    handler = {"register": cmd_register, "check": cmd_check, "run": cmd_run}[args.command]
    return handler(args)


def main() -> None:
    try:
        code = run(sys.argv[1:])
    except (ApiError, ValueError, OSError, RuntimeError, ImportError) as err:
        print(f"error: {err}", file=sys.stderr)
        code = EXIT_ERROR
    sys.exit(code)


if __name__ == "__main__":
    main()
