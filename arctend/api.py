"""A small client for the public Arctend API (standard library only).

Lifted from the Arctend starter bot (``starter-bot/python/arctend_starter/api.py``, same licence):
every error from the API is an RFC 9457 problem document with a stable ``code``, turned into an
``ApiError`` here. On HTTP 429 the client honours a short ``Retry-After`` once and then gives up
(daily limits reset at 00:00 UTC, so waiting hours makes no sense). Redirects are refused, so the
key never leaves the host it was meant for.

This module never imports ``forecasting_tools``: ``check`` and ``register`` work without it.
"""

from __future__ import annotations

import email.utils
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from typing import Any

VERSION = "0.2.0"
USER_AGENT = f"arctend-python/{VERSION}"

SANDBOX_API_URL = "https://sandbox.arctend.com"
PRODUCTION_API_URL = "https://api.arctend.com"
DEFAULT_API_URL = SANDBOX_API_URL
"""The practice sandbox, so a first run can never touch production by accident."""

MAX_RETRY_AFTER_SECONDS = 60
MAX_ATTEMPTS = 2
"""One retry after a short ``Retry-After``; a second 429 is final."""
REQUEST_TIMEOUT_SECONDS = 30

Json = dict[str, Any]


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, retry_after: int | None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.retry_after = retry_after


class RedirectRefusedError(ApiError):
    """A server answered with a redirect. Requests here carry an API key, so the client never
    follows one: the key would be sent to the new location."""

    def __init__(self, method: str, url: str, status: int, location: str | None) -> None:
        host = urllib.parse.urlsplit(location).hostname if location else None
        path = urllib.parse.urlsplit(url).path
        super().__init__(
            status,
            "redirect_refused",
            f"{method} {path}: HTTP {status} redirect to {host or 'an unknown host'} refused "
            "(a request carrying a key is never redirected)",
            None,
        )
        self.location_host = host


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        fp.close()
        raise RedirectRefusedError(req.get_method(), req.full_url, code, newurl)


_NO_REDIRECT_OPENER = urllib.request.build_opener(_RefuseRedirects())


def open_without_redirects(request: urllib.request.Request, timeout: float) -> Any:
    """Like ``urllib.request.urlopen``, but a 3xx answer is an error and is never followed, so a
    key in the request headers never reaches another location."""
    return _NO_REDIRECT_OPENER.open(request, timeout=timeout)


def parse_retry_after(value: str | None) -> int | None:
    """Retry-After is either a number of seconds or an HTTP date."""
    if value is None or not value.strip():
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        return max(0, math.ceil((when - datetime.now(UTC)).total_seconds()))
    if not math.isfinite(seconds):
        return None
    return max(0, math.ceil(seconds))


def idempotency_key(question_id: str, now: datetime | None = None) -> str:
    """One key per (question, UTC day): a retry on the same day can never create a second
    forecast (the scorer takes one snapshot a day anyway)."""
    day = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y-%m-%d")
    return f"arctend-ft:{question_id}:{day}"


def manual_idempotency_key(question_id: str, body: Json, now: datetime | None = None) -> str:
    """The key of a forecast submitted by hand (``arctend forecast``, ``submit_forecast``):
    ``arctend-cli:<question id>:<UTC date>:<16 hex characters of the body's SHA-256>``. The same
    values on the same day replay the first receipt; corrected values are a new forecast."""
    day = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y-%m-%d")
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"arctend-cli:{question_id}:{day}:{digest}"


VECTOR_SUM_TOLERANCE = 1e-6


def _check_probability(value: float, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{what} must be a number, not {value!r}")
    p = float(value)
    if not math.isfinite(p) or p < 0 or p > 1:
        raise ValueError(f"{what} must be a number from 0 to 1, not {value!r}")
    return p


def forecast_body(
    *,
    probability: float | None = None,
    vector: Sequence[float] | None = None,
    rationale: str | None = None,
) -> Json:
    """The body of ``PUT /v1/questions/{id}/forecast``: ``probability`` for a binary question or
    ``vector`` (one probability per option or range, in ``idx`` order, summing to 1) for
    multiple choice and numeric. Raises ``ValueError`` before anything is sent."""
    if (probability is None) == (vector is None):
        raise ValueError("give exactly one of probability (binary) and vector")
    body: Json
    if probability is not None:
        body = {"probability": _check_probability(probability, "the probability")}
    else:
        values = [_check_probability(v, "each vector entry") for v in vector or []]
        if len(values) < 2:
            raise ValueError("a vector needs at least two entries")
        total = math.fsum(values)
        if abs(total - 1) > VECTOR_SUM_TOLERANCE:
            raise ValueError(f"the vector must sum to 1, not {total:g}")
        body = {"vector": values}
    if rationale is not None:
        body["rationale"] = rationale
    return body


Opener = Callable[[urllib.request.Request, float], Any]


def _default_opener(request: urllib.request.Request, timeout: float) -> Any:
    return open_without_redirects(request, timeout)  # http/https only (checked in __init__)


def _quote(question_id: str) -> str:
    return urllib.parse.quote(question_id, safe="")


class ArctendClient:
    def __init__(
        self,
        base_url: str = DEFAULT_API_URL,
        api_key: str | None = None,
        opener: Opener = _default_opener,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("ARCTEND_API_URL must start with http:// or https://")
        self._api_key = api_key
        self._opener = opener
        self._sleep = sleep

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> ArctendClient:
        """``ARCTEND_API_URL`` (default the sandbox, ``https://sandbox.arctend.com``) and
        ``ARCTEND_API_KEY``. Nothing is read from or written to disk."""
        env = os.environ if environ is None else environ
        url = (env.get("ARCTEND_API_URL") or "").strip() or DEFAULT_API_URL
        key = (env.get("ARCTEND_API_KEY") or "").strip() or None
        return cls(url, key)

    @property
    def has_key(self) -> bool:
        return bool(self._api_key)

    def set_api_key(self, key: str) -> None:
        self._api_key = key

    def request(
        self,
        method: str,
        path: str,
        body: Any = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        attempt = 1
        while True:
            all_headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
            all_headers.update(headers or {})
            if self._api_key:
                all_headers["Authorization"] = f"Bearer {self._api_key}"
            data = None
            if body is not None:
                all_headers["Content-Type"] = "application/json"
                data = json.dumps(body).encode("utf-8")
            req = urllib.request.Request(
                self.base_url + path, data=data, headers=all_headers, method=method
            )
            try:
                with self._opener(req, REQUEST_TIMEOUT_SECONDS) as res:
                    text = res.read().decode("utf-8")
                    return json.loads(text) if text else None
            except urllib.error.HTTPError as err:
                text = err.read().decode("utf-8", errors="replace")
                retry_after = parse_retry_after(err.headers.get("Retry-After"))
                if (
                    err.code == 429
                    and retry_after is not None
                    and retry_after <= MAX_RETRY_AFTER_SECONDS
                    and attempt < MAX_ATTEMPTS
                ):
                    self._sleep(retry_after)
                    attempt += 1
                    continue
                code, detail = "http_error", text[:300]
                try:
                    problem = json.loads(text)
                    code = problem.get("code") or code
                    detail = problem.get("detail") or problem.get("title") or detail
                except (ValueError, AttributeError):
                    pass
                raise ApiError(
                    err.code, code, f"{method} {path}: {err.code} {code}: {detail}", retry_after
                ) from None

    def get_challenge(self) -> Json:
        return self.request("GET", "/v1/agents/register/challenge")

    def register(
        self,
        name: str,
        challenge: str,
        nonce: str,
        description: str | None = None,
        homepage: str | None = None,
        public_key: str | None = None,
        signature: str | None = None,
    ) -> Json:
        body: Json = {"name": name, "challenge": challenge, "nonce": nonce}
        for field, value in (
            ("description", description),
            ("homepage", homepage),
            ("public_key", public_key),
            ("signature", signature),
        ):
            if value is not None:
                body[field] = value
        return self.request("POST", "/v1/agents/register", body=body)

    def me(self) -> Json:
        return self.request("GET", "/v1/me")

    def questions_page(self, topic: str | None = None, cursor: str | None = None) -> Json:
        """One page of open public questions (``items``, ``next_cursor``)."""
        params = {"status": "open", "limit": "50"}
        if topic:
            params["topic"] = topic
        if cursor:
            params["cursor"] = cursor
        return self.request("GET", "/v1/questions?" + urllib.parse.urlencode(params))

    def open_questions(self, topic: str | None = None) -> Iterator[Json]:
        """Every open public question, following ``next_cursor`` page by page."""
        cursor: str | None = None
        while True:
            page = self.questions_page(topic, cursor)
            yield from page["items"]
            cursor = page.get("next_cursor")
            if not cursor:
                return

    def get_question(self, question_id: str) -> Json:
        return self.request("GET", f"/v1/questions/{_quote(question_id)}")

    def get_consensus(self, question_id: str) -> Json:
        return self.request("GET", f"/v1/questions/{_quote(question_id)}/consensus")

    def get_consensus_history(self, question_id: str) -> Json:
        """The first page of the median series' history, newest first (needs a key)."""
        return self.request("GET", f"/v1/questions/{_quote(question_id)}/consensus/history")

    def get_rating(self, agent_id: str) -> Json:
        return self.request("GET", f"/v1/agents/{_quote(agent_id)}/rating")

    def get_my_forecast(self, question_id: str) -> Json | None:
        """This agent's standing forecast on the question, or None if it has none."""
        try:
            return self.request("GET", f"/v1/questions/{_quote(question_id)}/forecast")
        except ApiError as err:
            if err.status == 404:
                return None
            raise

    def put_forecast(self, question_id: str, body: Json, idempotency_key: str) -> Json:
        return self.request(
            "PUT",
            f"/v1/questions/{_quote(question_id)}/forecast",
            body=body,
            headers={"Idempotency-Key": idempotency_key},
        )

    def submit_forecast(
        self,
        question_id: str,
        *,
        probability: float | None = None,
        vector: Sequence[float] | None = None,
        rationale: str | None = None,
    ) -> Json:
        """Submit one forecast by hand: ``probability`` for a binary question, or ``vector`` for
        multiple choice and numeric. The key is ``manual_idempotency_key``, so the same values
        on the same UTC day return the first receipt with ``"replayed": true``."""
        body = forecast_body(probability=probability, vector=vector, rationale=rationale)
        return self.put_forecast(question_id, body, manual_idempotency_key(question_id, body))

    def practice_results_page(self, cursor: str | None = None) -> Json:
        """One page of this agent's practice results, newest first (``items``, ``next_cursor``).
        The practice sandbox only (needs a key); production answers 404."""
        query = "?" + urllib.parse.urlencode({"cursor": cursor}) if cursor else ""
        return self.request("GET", "/v1/practice/results" + query)

    def practice_results(self) -> Iterator[Json]:
        """Every practice result of this agent, newest first, following ``next_cursor``."""
        cursor: str | None = None
        while True:
            page = self.practice_results_page(cursor)
            yield from page["items"]
            cursor = page.get("next_cursor")
            if not cursor:
                return
