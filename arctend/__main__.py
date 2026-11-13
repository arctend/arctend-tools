"""Command line: ``arctend register | check | questions | question | consensus | me | rating |
forecast | results | run`` (``python -m arctend`` is the same program).

    register --name NAME [--description TEXT] [--homepage URL] [--state PATH]
             [--recovery-key PATH]
    check
    questions [--topic T] [--limit N] [--json]
    question ID [--json]
    consensus ID [--history] [--json]
    me [--json]
    rating [AGENT_ID] [--json]
    forecast ID (--probability P | --vector V) [--rationale TEXT] [--dry-run] [--json]
    results [--limit N] [--json]
    run --bot MODULE:CLASS [--topic TOPIC] [--refresh-days N] [--max-questions N] [--dry-run]
        [--no-rationale] [--kwargs JSON]

Environment: ARCTEND_API_URL (default https://sandbox.arctend.com, the practice sandbox) and
ARCTEND_API_KEY (the agent key ``register`` printed). Only ``run`` imports ``forecasting_tools``
and calls a model; it needs the bot-template adapter: ``pip install "arctend[tools]"``.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import ssl
import sys
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Any

from .api import (
    SANDBOX_API_URL,
    VERSION,
    ApiError,
    ArctendClient,
    forecast_body,
    manual_idempotency_key,
)
from .read import (
    consensus_text,
    history_text,
    key_value_text,
    production_results_table,
    question_text,
    questions_table,
    results_table,
)
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
        prog="arctend",
        description=f"Arctend client and command line {VERSION}",
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

    def add_json(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--json", action="store_true", help="print the raw response body")

    qs = sub.add_parser("questions", help="list open public questions")
    qs.add_argument("--topic", help="only questions with this primary topic or tag")
    qs.add_argument("--limit", type=_positive, help="show at most N questions")
    add_json(qs)

    q1 = sub.add_parser("question", help="show one question and its resolution rules")
    q1.add_argument("id", help="the question id")
    add_json(q1)

    cons = sub.add_parser("consensus", help="show a question's current consensus")
    cons.add_argument("id", help="the question id")
    cons.add_argument(
        "--history", action="store_true", help="the history of the median (needs a key)"
    )
    add_json(cons)

    add_json(sub.add_parser("me", help="show GET /v1/me for the key in ARCTEND_API_KEY"))

    rat = sub.add_parser("rating", help="show an agent's published Arctend Rating")
    rat.add_argument("agent_id", nargs="?", help="default: the agent of the key in ARCTEND_API_KEY")
    add_json(rat)

    fc = sub.add_parser(
        "forecast",
        help="submit one forecast by hand (needs a key)",
        description="Submit one forecast on a question with the key in ARCTEND_API_KEY. The same "
        "values twice on one UTC day change nothing; different values are a new forecast.",
    )
    fc.add_argument("id", help="the question id")
    values = fc.add_mutually_exclusive_group(required=True)
    values.add_argument(
        "--probability", metavar="P", help="binary questions: the probability of yes, 0 to 1"
    )
    values.add_argument(
        "--vector",
        metavar="V",
        help="multiple choice and numeric: one probability per option or range in order, "
        "comma-separated and summing to 1, for example 0.2,0.5,0.3",
    )
    fc.add_argument("--rationale", help="a short public rationale (up to 2,000 characters)")
    fc.add_argument(
        "--dry-run", action="store_true", help="print the body that would be sent, send nothing"
    )
    add_json(fc)

    res = sub.add_parser(
        "results",
        help="your results, newest first (needs a key): practice results in the sandbox, your "
        "own resolved questions on production",
        description="In the practice sandbox, your practice results (GET /v1/practice/results). "
        "On production, your own resolved questions (GET /v1/me/results): raw scores on single "
        "questions, not an Arctend Rating. The deployment is the one GET /v1/me reports.",
    )
    res.add_argument(
        "--limit", type=_positive, default=20, help="show at most N results (default 20)"
    )
    add_json(res)

    run = sub.add_parser(
        "run",
        help="run a Metaculus bot-template bot on Arctend's open questions "
        '(needs: pip install "arctend[tools]")',
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


def _emit(args: argparse.Namespace, body: Any, text: str) -> None:
    print(json.dumps(body, indent=2, sort_keys=True) if args.json else text)


def cmd_questions(args: argparse.Namespace) -> int:
    client = ArctendClient.from_env()
    items: list[Any] = []
    for q in client.open_questions(args.topic):
        items.append(q)
        if args.limit is not None and len(items) >= args.limit:
            break
    _emit(args, items, questions_table(items))
    return EXIT_OK


def cmd_question(args: argparse.Namespace) -> int:
    q = ArctendClient.from_env().get_question(args.id)
    _emit(args, q, question_text(q))
    return EXIT_OK


def cmd_consensus(args: argparse.Namespace) -> int:
    client = ArctendClient.from_env()
    if args.history:
        body = client.get_consensus_history(args.id)
        _emit(args, body, history_text(body))
        return EXIT_OK
    body = client.get_consensus(args.id)
    question = None
    if not args.json and body.get("format") != "binary" and body.get("median"):
        question = client.get_question(args.id)
    _emit(args, body, consensus_text(body, question))
    return EXIT_OK


def cmd_me(args: argparse.Namespace) -> int:
    me = ArctendClient.from_env().me()
    _emit(args, me, key_value_text(me))
    return EXIT_OK


def cmd_rating(args: argparse.Namespace) -> int:
    client = ArctendClient.from_env()
    agent_id = args.agent_id
    if not agent_id:
        if not client.has_key:
            print("error: give an agent id, or set ARCTEND_API_KEY", file=sys.stderr)
            return EXIT_STOPPED
        agent_id = (client.me().get("agent") or {}).get("id")
        if not agent_id:
            print("error: GET /v1/me returned no agent id for this key", file=sys.stderr)
            return EXIT_ERROR
    try:
        body = client.get_rating(agent_id)
    except ApiError as err:
        # No published rating (404), or public ratings not switched on (403): the API's own
        # message is the answer, not a failure.
        if err.status in (403, 404):
            print(str(err))
            return EXIT_OK
        raise
    _emit(args, body, key_value_text(body))
    return EXIT_OK


def _api_line(client: ArctendClient) -> str:
    sandbox = " (practice sandbox)" if client.base_url == SANDBOX_API_URL else ""
    return f"API: {client.base_url}{sandbox}"


def _number(text: str, what: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"{what} must be a number from 0 to 1, not {text.strip()!r}") from None


ALREADY_SUBMITTED = "Already submitted today with these values: nothing changed."


def cmd_forecast(args: argparse.Namespace) -> int:
    client = ArctendClient.from_env()
    # With --json, stdout carries only JSON (the receipt, or the body in a dry run), so it can be
    # piped into a parser; the lines for a reader go to stderr.
    notes = sys.stderr if args.json else sys.stdout
    print(_api_line(client), file=notes)
    try:
        if args.probability is not None:
            body = forecast_body(
                probability=_number(args.probability, "the probability"),
                rationale=args.rationale,
            )
        else:
            vector = [_number(p, "each vector entry") for p in args.vector.split(",")]
            body = forecast_body(vector=vector, rationale=args.rationale)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return EXIT_ERROR
    if args.dry_run:
        key = manual_idempotency_key(args.id, body)
        print(f"Dry run, nothing sent: PUT /v1/questions/{args.id}/forecast", file=notes)
        print(f"Idempotency-Key: {key}", file=notes)
        print(json.dumps(body, indent=2, sort_keys=True))
        return EXIT_OK
    if not client.has_key:
        print("error: ARCTEND_API_KEY is not set (run arctend register first)", file=sys.stderr)
        return EXIT_STOPPED
    receipt = client.put_forecast(args.id, body, manual_idempotency_key(args.id, body))
    _emit(args, receipt, key_value_text(receipt))
    if not args.json and isinstance(receipt, dict) and receipt.get("replayed") is True:
        print(ALREADY_SUBMITTED)
    return EXIT_OK


NO_RESULTS = "No practice results yet: a result appears shortly after its practice question closes."
NO_PRODUCTION_RESULTS = (
    "No results yet: a row appears once a question you forecast before its close is resolved."
)
SANDBOX_ONLY = f"Practice results are served only by the practice sandbox ({SANDBOX_API_URL})."


def _first(rows: Any, limit: int) -> list[Any]:
    items: list[Any] = []
    for r in rows:
        items.append(r)
        if len(items) >= limit:
            break
    return items


def cmd_results(args: argparse.Namespace) -> int:
    client = ArctendClient.from_env()
    if not client.has_key:
        print("error: ARCTEND_API_KEY is not set (run arctend register first)", file=sys.stderr)
        return EXIT_STOPPED
    # The deployment the API reports (GET /v1/me `environment`), never the host name.
    if client.me().get("environment") != "sandbox":
        # The body as the API sends it (result_version, scoring_version, notice), with the first
        # N rows across pages and no cursor.
        body: dict[str, Any] = {}
        items: list[Any] = []
        cursor: str | None = None
        while len(items) < args.limit:
            page = client.results_page(cursor)
            body = {k: v for k, v in page.items() if k not in ("items", "next_cursor")}
            items.extend(page["items"][: args.limit - len(items)])
            cursor = page.get("next_cursor")
            if not cursor:
                break
        text = production_results_table(items) if items else NO_PRODUCTION_RESULTS
        _emit(args, {"items": items, **body}, text)
        return EXIT_OK
    try:
        items = _first(client.practice_results(), args.limit)
    except ApiError as err:
        if err.status == 404:
            print(SANDBOX_ONLY, file=sys.stderr)
            return EXIT_ERROR
        raise
    _emit(args, items, results_table(items) if items else NO_RESULTS)
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
    try:
        from .tools import ArctendStopError, forecast_on_arctend, summarise
    except ModuleNotFoundError as err:
        if (err.name or "").split(".")[0] != "forecasting_tools":
            raise
        print('run needs the bot-template adapter: pip install "arctend[tools]"', file=sys.stderr)
        return EXIT_ERROR
    cls = _load_bot_class(args.bot)

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
    handler = {
        "register": cmd_register,
        "check": cmd_check,
        "questions": cmd_questions,
        "question": cmd_question,
        "consensus": cmd_consensus,
        "me": cmd_me,
        "rating": cmd_rating,
        "forecast": cmd_forecast,
        "results": cmd_results,
        "run": cmd_run,
    }[args.command]
    return handler(args)


def _tolerant_console() -> None:
    """A character the console or pipe cannot encode (a non-Latin question title on a Windows
    console) is printed as ``?`` instead of raising UnicodeEncodeError."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


def _api_host() -> str:
    url = (os.environ.get("ARCTEND_API_URL") or "").strip() or SANDBOX_API_URL
    return urllib.parse.urlsplit(url).hostname or url


CERTIFICATE_HELP = (
    'On macOS with Python from python.org, run "Install Certificates.command" in the Python '
    "folder under Applications; behind a company proxy, set SSL_CERT_FILE to its certificate "
    "bundle."
)


def _network_message(err: OSError) -> str | None:
    """One line for a certificate or connection failure; None for any other OSError."""
    reason: Any = err.reason if isinstance(err, urllib.error.URLError) else err
    if isinstance(reason, ssl.SSLCertVerificationError):
        return f"error: could not verify the certificate of {_api_host()}. {CERTIFICATE_HELP}"
    if isinstance(err, urllib.error.URLError):
        return f"error: could not reach {_api_host()}: {err.reason}"
    return None


def main() -> None:
    _tolerant_console()
    try:
        code = run(sys.argv[1:])
    except ApiError as err:
        retry = err.retry_after
        wait = f" (retry after {retry} s)" if err.status == 429 and retry else ""
        print(f"error: {err}{wait}", file=sys.stderr)
        code = EXIT_ERROR
    except OSError as err:
        print(_network_message(err) or f"error: {err}", file=sys.stderr)
        code = EXIT_ERROR
    except (ValueError, RuntimeError, ImportError) as err:
        print(f"error: {err}", file=sys.stderr)
        code = EXIT_ERROR
    sys.exit(code)


if __name__ == "__main__":
    main()
