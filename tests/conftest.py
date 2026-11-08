"""Shared test fixtures: Arctend question builders, a fake Arctend HTTP server and a stub bot.

No network, no model key and no Metaculus token: litellm (imported by forecasting_tools) is told
to use its bundled model cost map instead of fetching one, the fake server listens on 127.0.0.1,
and the stub bot never calls a model.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import threading
import urllib.parse
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.pop("METACULUS_TOKEN", None)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
from forecasting_tools import (  # noqa: E402
    BinaryQuestion,
    ForecastBot,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOption,
    PredictedOptionList,
    ReasonedPrediction,
)

Json = dict[str, Any]

ISO = "%Y-%m-%dT%H:%M:%S.000Z"


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime(ISO)


def make_question(fmt: str, n: int = 1, **overrides: Any) -> Json:
    """A full Arctend question object with every required field of the contract."""
    now = datetime.now(UTC)
    qid = str(uuid.UUID(int=n))
    q: Json = {
        "id": qid,
        "slug": f"question-{n}",
        "version": 1,
        "title": f"Question {n}?",
        "format": fmt,
        "topic_primary": "macro",
        "tags": ["macro", "rates"],
        "cluster_id": None,
        "open_at": iso(now - timedelta(days=1)),
        "close_at": iso(now + timedelta(days=10)),
        "resolve_at": iso(now + timedelta(days=11)),
        "status": "open",
        "baseline_policy": "standard",
        "visibility": "public",
        "author_type": "house",
        "k": 2,
        "template_id": None,
        "benchmark_track": "none",
        "registry_number": f"Q-2026-{n:05d}",
        "final_outcome": None,
        "final_at": None,
        "publication": None,
        "summoned": False,
        "body": f"Background for question {n}.",
        "resolution_spec": {
            "criteria": "YES if the published rate is above 4.",
            "unit": "percent",
            "edge_cases": {"postponed": "Waits for the publication", "cancelled": "Annulled"},
            "comparison": {"op": ">", "threshold": 4, "rounding": 2},
            "primary_source": {"name": "Central bank", "url": "https://example.org/rates"},
            "backup_source": {"name": "Statistics office", "url": "https://example.net/stats"},
        },
        "author_id": None,
        "provenance": {"generator": "test"},
        "options": [],
        "bins": [],
        "clarifications": [],
        "created_at": iso(now - timedelta(days=2)),
        "updated_at": iso(now - timedelta(days=2)),
    }
    if fmt == "binary":
        q["k"] = 2
    elif fmt == "multiple_choice":
        q["options"] = [
            {"idx": 2, "label": "Green"},
            {"idx": 0, "label": "Red"},
            {"idx": 1, "label": "Blue"},
        ]
        q["k"] = 3
    elif fmt == "numeric":
        q["bins"] = five_bins()
        q["k"] = 5
    q.update(overrides)
    return q


def five_bins() -> list[Json]:
    edges = [None, 0.0, 10.0, 30.0, 60.0, None]
    return [
        {
            "idx": i,
            "lower": edges[i],
            "upper": edges[i + 1],
            "lower_open": edges[i] is None,
            "upper_open": edges[i + 1] is None,
        }
        for i in range(5)
    ]


def two_bins() -> list[Json]:
    return [
        {"idx": 0, "lower": None, "upper": 50.0, "lower_open": True, "upper_open": False},
        {"idx": 1, "lower": 50.0, "upper": None, "lower_open": False, "upper_open": True},
    ]


def summary_of(q: Json) -> Json:
    keys = (
        "id slug version title format topic_primary tags cluster_id open_at close_at resolve_at "
        "status baseline_policy visibility author_type k template_id benchmark_track "
        "registry_number final_outcome final_at publication summoned"
    ).split()
    return {k: q[k] for k in keys if k in q}


# --- Fake Arctend server ----------------------------------------------------------------------


class FakeArctend:
    """Enough of the Arctend API for the adapter: me, the question list (paginated), full
    questions, standing forecasts, PUT with idempotency keys, and registration. ``script`` queues
    forced answers per ``(method, path)``."""

    def __init__(self) -> None:
        self.api_key = "atk_test.secret"
        self.questions: dict[str, Json] = {}
        self.standing: dict[str, Json] = {}
        self.idempotency: dict[str, tuple[Json, Json]] = {}
        self.requests: list[Json] = []
        self.script: dict[tuple[str, str], list[tuple[int, dict[str, str], Json]]] = {}
        self.page_size = 2
        self.sandbox = False
        self.difficulty = 4
        self.url = ""
        self.lock = threading.Lock()

    def add(self, q: Json, standing_days_ago: float | None = None) -> Json:
        self.questions[q["id"]] = q
        if standing_days_ago is not None:
            received = datetime.now(UTC) - timedelta(days=standing_days_ago)
            self.standing[q["id"]] = {
                "forecast_id": str(uuid.uuid4()),
                "question_id": q["id"],
                "received_at": iso(received),
            }
        return q

    def puts(self) -> list[Json]:
        return [r for r in self.requests if r["method"] == "PUT"]

    def paths(self) -> list[str]:
        return [r["path"] for r in self.requests]

    def handle(self, method: str, raw_path: str, headers: Any, body: Json | None) -> tuple:
        parsed = urllib.parse.urlsplit(raw_path)
        path, query = parsed.path, urllib.parse.parse_qs(parsed.query)
        with self.lock:
            self.requests.append(
                {
                    "method": method,
                    "path": path,
                    "query": query,
                    "headers": dict(headers),
                    "body": body,
                }
            )
            forced = self.script.get((method, path))
            if forced:
                return forced.pop(0)
        if path == "/v1/agents/register/challenge":
            return 200, {}, {"challenge": "test-challenge", "difficulty": self.difficulty}
        if path == "/v1/agents/register" and method == "POST":
            return 201, {}, self._registration(body or {})
        if headers.get("Authorization") != f"Bearer {self.api_key}":
            return 401, {}, {"code": "invalid_key", "title": "Unauthorized"}
        if path == "/v1/me":
            return (
                200,
                {},
                {
                    "tier": "provisional",
                    "limits": {"forecast_writes_per_day": 200, "reads_per_day": 300},
                    "agent": {"slug": "test-bot", "name": "Test bot", "is_provisional": True},
                },
            )
        if path == "/v1/questions":
            return 200, {}, self._page(query)
        parts = path.split("/")
        if len(parts) >= 4 and parts[1:3] == ["v1", "questions"]:
            qid = urllib.parse.unquote(parts[3])
            if qid not in self.questions:
                return 404, {}, {"code": "not_found"}
            if len(parts) == 4:
                return 200, {}, copy.deepcopy(self.questions[qid])
            if parts[4] == "forecast" and method == "GET":
                if qid in self.standing:
                    return 200, {}, self.standing[qid]
                return 404, {}, {"code": "not_found", "title": "No standing forecast"}
            if parts[4] == "forecast" and method == "PUT":
                return self._put(qid, headers.get("Idempotency-Key"), body or {})
        return 404, {}, {"code": "not_found"}

    def _page(self, query: dict[str, list[str]]) -> Json:
        items = [summary_of(q) for q in self.questions.values()]
        start = int(query.get("cursor", ["0"])[0])
        page = items[start : start + self.page_size]
        nxt = start + self.page_size
        return {"items": page, "next_cursor": str(nxt) if nxt < len(items) else None}

    def _put(self, qid: str, key: str | None, body: Json) -> tuple:
        if key and key in self.idempotency:
            stored_body, receipt = self.idempotency[key]
            if stored_body != body:
                return 409, {}, {"code": "idempotency_conflict"}
            return 200, {}, {**receipt, "replayed": True}
        receipt = {
            "forecast_id": str(uuid.uuid4()),
            "question_id": qid,
            "agent_id": "agent-1",
            "received_at": iso(datetime.now(UTC)),
            "content_hash": "0" * 64,
            "vector": body.get("vector"),
            "probability": body.get("probability"),
            "replayed": False,
        }
        if key:
            self.idempotency[key] = (body, receipt)
        self.standing[qid] = {
            "forecast_id": receipt["forecast_id"],
            "question_id": qid,
            "received_at": receipt["received_at"],
        }
        return 201, {}, receipt

    def _registration(self, body: Json) -> Json:
        reg: Json = {
            "agent": {
                "id": "00000000-0000-4000-8000-0000000000aa",
                "slug": "my-bot-x1",
                "name": body.get("name"),
            },
            "api_key": {"token": "atk_fresh_secret_token_123456", "prefix": "atk_fres"},
            "terms": "https://api.arctend.com/docs/terms.md",
        }
        if self.sandbox:
            reg["practice"] = True
        else:
            reg["claim_code"] = "claim-code-secret-987"
            reg["claim"] = {
                "endpoint": f"{self.url}/v1/agents/claim",
                "how": "A verified developer calls POST /v1/agents/claim.",
            }
        if body.get("public_key"):
            reg["recovery_public_key"] = body["public_key"]
        return reg


def _start(fake: FakeArctend) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def _answer(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            body = json.loads(raw) if raw else None
            status, headers, payload = fake.handle(self.command, self.path, self.headers, body)
            data = json.dumps(payload).encode()
            self.send_response(status)
            ctype = "application/json" if status < 400 else "application/problem+json"
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            for k, v in headers.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        do_GET = _answer  # noqa: N815
        do_PUT = _answer  # noqa: N815
        do_POST = _answer  # noqa: N815

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    fake.url = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def fake() -> Iterator[FakeArctend]:
    f = FakeArctend()
    server = _start(f)
    try:
        yield f
    finally:
        server.shutdown()
        server.server_close()


# --- Stub bot ---------------------------------------------------------------------------------


class StubBot(ForecastBot):
    """Fixed predictions, no research, no model. ``fail_titles`` raise for those questions."""

    binary_value = 0.62
    option_probabilities: dict[str, float] = {"Red": 0.5, "Blue": 0.3, "Green": 0.2}

    def __init__(self, fail_titles: tuple[str, ...] = (), **kwargs: Any) -> None:
        kwargs.setdefault("enable_summarize_research", False)
        super().__init__(**kwargs)
        self.fail_titles = fail_titles
        self.seen: list[MetaculusQuestion] = []

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, Any]:
        return {"default": None, "summarizer": None, "researcher": None, "parser": None}

    async def run_research(self, question: MetaculusQuestion) -> str:
        self.seen.append(question)
        if question.question_text in self.fail_titles:
            raise RuntimeError(f"research failed for {question.question_text}")
        return ""

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        return ReasonedPrediction(prediction_value=self.binary_value, reasoning="Base rate.")

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        probs = self.option_probabilities
        if set(question.options) != set(probs):
            probs = {o: 1 / len(question.options) for o in question.options}
        options = [PredictedOption(option_name=o, probability=p) for o, p in probs.items()]
        return ReasonedPrediction(
            prediction_value=PredictedOptionList(predicted_options=options), reasoning="Counts."
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        return ReasonedPrediction(
            prediction_value=uniform_distribution(question), reasoning="Spread evenly."
        )


def uniform_distribution(question: NumericQuestion) -> NumericDistribution:
    """A CDF that rises linearly from 0 at lower_bound to 1 at upper_bound (no tail mass)."""
    lo, hi = question.lower_bound, question.upper_bound
    points = [Percentile(value=lo + (hi - lo) * i / 200, percentile=i / 200) for i in range(201)]
    return NumericDistribution(
        declared_percentiles=points,
        open_upper_bound=question.open_upper_bound,
        open_lower_bound=question.open_lower_bound,
        upper_bound=hi,
        lower_bound=lo,
        zero_point=None,
        cdf_size=201,
        standardize_cdf=False,
    )


class ExplodingMetaculusClient:
    """Raises on any use: proves the runner never touches Metaculus."""

    touched = False

    def __getattr__(self, name: str) -> Any:
        ExplodingMetaculusClient.touched = True
        raise AssertionError(f"MetaculusClient.{name} was used")
