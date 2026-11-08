"""The runner: Arctend's open questions through an unchanged ``forecasting_tools`` bot.

``forecast_on_arctend`` lists Arctend's open questions, maps them to the question classes the bot
already consumes, calls ``bot.forecast_questions`` and posts each prediction back. It uses only
the ``ForecastBot`` interface, never constructs a ``MetaculusClient``, never reads
``METACULUS_TOKEN``, and switches the bot's Metaculus publication off for the duration of the call.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from forecasting_tools import ForecastBot

from .api import ApiError, ArctendClient, idempotency_key
from .mapping import (
    MappingError,
    default_agent_version,
    to_arctend_submission,
    to_forecasting_tools_question,
)

Json = dict[str, Any]
Status = Literal["submitted", "dry_run", "skipped", "error", "not_submitted"]

DAY_SECONDS = 86_400


@dataclass(frozen=True)
class ArctendRunResult:
    question_id: str
    status: Status
    slug: str | None = None
    body: Json | None = None
    """The submission body (``submitted`` and ``dry_run``)."""
    receipt: Json | None = None
    """The API's receipt (``submitted``)."""
    error: str | None = None
    """Why the question failed or was not submitted."""
    detail: str | None = None
    """Why the question was skipped."""


class ArctendStopError(RuntimeError):
    """The run cannot start (for example, the key was rejected)."""


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _is_fatal_status(status: int) -> bool:
    """After these every later request would fail too: a redirect (never followed with a key),
    a rejected key, a forbidden action or a server error."""
    return 300 <= status < 400 or status in (401, 403) or status >= 500


def _describe(body: Json) -> str:
    if "probability" in body:
        return f"P(yes) = {body['probability']:.3f}"
    return "[" + ", ".join(f"{x:.3f}" for x in body["vector"]) + "]"


def _check_identity(client: ArctendClient) -> Json:
    if not client.has_key:
        raise ArctendStopError(
            "ARCTEND_API_KEY is not set. Register first: python -m arctend_forecasting_tools "
            'register --name "My bot"'
        )
    try:
        me = client.me()
    except ApiError as err:
        if err.status in (401, 403):
            raise ArctendStopError(
                f"the Arctend key was rejected by {client.base_url} ({err}). Check "
                "ARCTEND_API_KEY, and that ARCTEND_API_URL is the deployment the key was issued "
                "by (sandbox keys start atk_sbx_ and work only in the sandbox)."
            ) from None
        raise
    if not me.get("agent"):
        raise ArctendStopError(
            "this key belongs to a developer account, not an agent: use the agent's own key"
        )
    limits = me.get("limits") or {}
    print(
        f"Agent {me['agent'].get('slug')} at {client.base_url}, tier {me.get('tier')}, "
        f"{limits.get('forecast_writes_per_day', '?')} forecasts per day."
    )
    return me


async def forecast_on_arctend(
    bot: ForecastBot,
    client: ArctendClient,
    *,
    topic: str | None = None,
    refresh_days: int = 7,
    max_questions: int | None = None,
    dry_run: bool = False,
    include_rationale: bool = True,
    agent_version: str | None = None,
) -> list[ArctendRunResult]:
    """Forecast Arctend's open questions with ``bot`` and submit the predictions.

    A question this agent forecast less than ``refresh_days`` days ago is marked
    ``already_forecasted``; the bot's own ``skip_previously_forecasted_questions`` decides
    whether it is forecast again. ``max_questions`` caps the batch after that skip. With
    ``dry_run`` the bot still runs (and still costs model calls) but nothing is submitted.
    """
    if refresh_days < 0:
        raise ValueError("refresh_days must be >= 0")
    if max_questions is not None and max_questions < 1:
        raise ValueError("max_questions must be >= 1")
    _check_identity(client)

    results: list[ArctendRunResult] = []
    batch: list[tuple[Json, Any]] = []
    skip_flag = bool(getattr(bot, "skip_previously_forecasted_questions", False))
    now = datetime.now(UTC)
    for summary in client.open_questions(topic):
        if max_questions is not None and len(batch) >= max_questions:
            break
        if _parse_time(summary["close_at"]) <= now:
            continue
        qid = summary["id"]
        standing = client.get_my_forecast(qid)
        fresh = bool(
            standing
            and (now - _parse_time(standing["received_at"])).total_seconds()
            < refresh_days * DAY_SECONDS
        )
        if fresh and skip_flag:
            # The bot would drop it in forecast_questions (skip_previously_forecasted_questions);
            # deciding here saves the full-question read (provisional keys read 300 a day).
            detail = f"standing forecast from {standing['received_at']}"
            print(f"skip {summary.get('slug', qid)}: {detail}")
            results.append(ArctendRunResult(qid, "skipped", summary.get("slug"), detail=detail))
            continue
        question = client.get_question(qid)
        try:
            ft_question = to_forecasting_tools_question(question, client.base_url)
        except MappingError as err:
            print(f"error {summary.get('slug', qid)}: {err}")
            results.append(ArctendRunResult(qid, "error", summary.get("slug"), error=str(err)))
            continue
        ft_question.already_forecasted = fresh
        batch.append((question, ft_question))

    if not batch:
        print("No questions to forecast.")
        return results

    print(f"Forecasting {len(batch)} question(s) with {type(bot).__name__}...")
    previous = getattr(bot, "publish_reports_to_metaculus", False)
    try:
        bot.publish_reports_to_metaculus = False
        reports = await bot.forecast_questions([ft for _, ft in batch], return_exceptions=True)
    finally:
        bot.publish_reports_to_metaculus = previous

    version = agent_version or default_agent_version(type(bot).__name__)
    results.extend(
        _submit_all(client, _pair(batch, reports), dry_run, include_rationale, version, now)
    )
    return results


def _pair(batch: list[tuple[Json, Any]], reports: Sequence[Any]) -> list[tuple[Json, Any]]:
    """Pair each report with its Arctend question. ``forecast_questions`` returns one result per
    question in order; if a subclass returns something else, pair reports by the question they
    carry and report the rest as missing."""
    if len(reports) == len(batch):
        return [(q, r) for (q, _), r in zip(batch, reports, strict=True)]
    by_id = {q["id"]: q for q, _ in batch}
    paired: list[tuple[Json, Any]] = []
    seen: set[str] = set()
    for r in reports:
        qid = None
        if not isinstance(r, BaseException):
            qid = (getattr(getattr(r, "question", None), "api_json", None) or {}).get("id")
        if qid in by_id and qid not in seen:
            paired.append((by_id[qid], r))
            seen.add(qid)
    for q, _ in batch:
        if q["id"] not in seen:
            paired.append((q, RuntimeError("the bot returned no report for this question")))
    return paired


def _submit_all(
    client: ArctendClient,
    paired: list[tuple[Json, Any]],
    dry_run: bool,
    include_rationale: bool,
    agent_version: str,
    now: datetime,
) -> list[ArctendRunResult]:
    results: list[ArctendRunResult] = []
    stop_reason: str | None = None
    for question, report in paired:
        qid, slug = question["id"], question.get("slug")
        label = slug or qid
        if stop_reason is not None:
            results.append(ArctendRunResult(qid, "not_submitted", slug, error=stop_reason))
            continue
        if isinstance(report, BaseException):
            message = f"{type(report).__name__}: {report}"[:500]
            print(f"error {label}: the bot raised {message}")
            results.append(ArctendRunResult(qid, "error", slug, error=message))
            continue
        try:
            body = to_arctend_submission(
                question, report, include_rationale=include_rationale, agent_version=agent_version
            )
        except MappingError as err:
            print(f"error {label}: {err}")
            results.append(ArctendRunResult(qid, "error", slug, error=str(err)))
            continue
        if dry_run:
            print(f"dry-run {label}: {body}")
            results.append(ArctendRunResult(qid, "dry_run", slug, body=body))
            continue
        try:
            receipt = client.put_forecast(qid, body, idempotency_key(qid, now))
        except ApiError as err:
            if err.status == 429 or _is_fatal_status(err.status):
                stop_reason = f"run stopped: {err}"
                print(f"stopping: {err}")
                results.append(ArctendRunResult(qid, "not_submitted", slug, error=str(err)))
                continue
            print(f"error {label}: {err}")
            results.append(ArctendRunResult(qid, "error", slug, error=str(err)))
            continue
        print(f"forecast {label}: {_describe(body)} (receipt {receipt.get('forecast_id')})")
        results.append(ArctendRunResult(qid, "submitted", slug, body=body, receipt=receipt))
    return results


def summarise(results: Sequence[ArctendRunResult]) -> str:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    labels = {
        "submitted": "submitted",
        "dry_run": "computed",
        "skipped": "skipped",
        "error": "errors",
        "not_submitted": "not submitted",
    }
    return ", ".join(f"{counts[s]} {label}" for s, label in labels.items() if counts.get(s))
