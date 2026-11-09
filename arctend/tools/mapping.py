"""Pure, deterministic mappings between Arctend questions and ``forecasting_tools`` objects.

- ``to_forecasting_tools_question``: the full ``GET /v1/questions/{id}`` object to a
  ``BinaryQuestion``, ``MultipleChoiceQuestion`` or ``NumericQuestion``.
- ``to_arctend_submission``: a ``ForecastReport`` back to the body of
  ``PUT /v1/questions/{id}/forecast``.

Design rules (ADR 0055): the Arctend consensus is never passed to the bot
(``community_prediction_at_access_time`` is always None); a numeric question with fewer than two
distinct finite bin edges is presented as multiple choice over its bins; ``id_of_post`` and
``id_of_question`` stay None, so nothing here can address a Metaculus question.

``forecasting_tools`` CDF convention (verified against the installed package, 0.3.5):
``NumericDistribution.get_cdf()`` returns ``cdf_size`` (201) ``Percentile(value, percentile)``
points, ``percentile`` on a 0 to 1 scale, ``value`` evenly spaced from exactly ``lower_bound``
(first point) to exactly ``upper_bound`` (last point) for a linear question;
``cdf[0].percentile`` is P(outcome < lower_bound) and ``cdf[-1].percentile`` is
P(outcome <= upper_bound).
"""

from __future__ import annotations

import math
import urllib.parse
from bisect import bisect_right
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from forecasting_tools import (
    BinaryQuestion,
    ForecastReport,
    MultipleChoiceQuestion,
    NumericQuestion,
)

from ..api import PRODUCTION_API_URL, VERSION

Json = dict[str, Any]

RATIONALE_MAX = 2000
"""The API's limit, counted in UTF-16 code units (JavaScript string length)."""
AGENT_VERSION_MAX = 64
CDF_SIZE = 201

PRESENTED_AS_KEY = "arctend_presented_as"

# Keys of ``resolution_spec`` that are never rendered into the prompt-facing text:
# - the sources get their own line at the end;
# - ``true_probabilities`` is the practice sandbox's answer key (the probabilities the practice
#   result compares against); putting it in the prompt would make practice feedback meaningless.
# Every key stays reachable through ``api_json``.
_SOURCE_KEYS = ("primary_source", "backup_source", "tiebreak_source")
_NOT_RENDERED = frozenset({*_SOURCE_KEYS, "true_probabilities"})


class MappingError(ValueError):
    """The question or the prediction cannot be mapped: skip this question and report it."""


# --- Arctend question -> forecasting_tools question -----------------------------------------


def _number(x: float) -> str:
    # Same text as JavaScript's String(number) for the values bins use (2.0 -> "2").
    return str(int(x)) if float(x).is_integer() else repr(float(x))


def describe_bin(b: Json) -> str:
    """The starter bot's bin text: ``(-infinity, 50)``, ``[50, +infinity)``."""
    lower = "(-infinity" if b["lower"] is None else f"[{_number(b['lower'])}"
    upper = "+infinity)" if b["upper"] is None else f"{_number(b['upper'])})"
    return f"{lower}, {upper}"


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise MappingError(f"time {value!r} has no timezone")
    return parsed


def _scalar(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int | float):
        return _number(value) if math.isfinite(value) else str(value)
    return str(value)


def _render(value: Any, key: str, indent: int, lines: list[str]) -> None:
    pad = "  " * indent
    label = key.replace("_", " ")
    if isinstance(value, dict):
        lines.append(f"{pad}{label}:")
        for k, v in value.items():
            _render(v, str(k), indent + 1, lines)
    elif isinstance(value, list):
        if all(not isinstance(v, dict | list) for v in value):
            lines.append(f"{pad}{label}: {', '.join(_scalar(v) for v in value)}")
        else:
            lines.append(f"{pad}{label}:")
            for item in value:
                if isinstance(item, dict | list):
                    _render(item, "-", indent + 1, lines)
                else:
                    lines.append(f"{pad}  - {_scalar(item)}")
    else:
        lines.append(f"{pad}{label}: {_scalar(value)}")


def _describe_source(source: Any) -> str:
    if not isinstance(source, dict):
        return _scalar(source)
    name = source.get("name")
    url = source.get("url")
    if name and url:
        return f"{name} ({url})"
    return str(name or url or "unnamed source")


def render_resolution_criteria(spec: Json | None) -> str:
    """``resolution_spec`` as readable lines: ``criteria`` first, then every other key in order,
    one ``key: value`` per line with nested objects indented; then one line naming the primary
    and backup sources. Never raw JSON."""
    spec = spec or {}
    lines: list[str] = []
    if "criteria" in spec:
        _render(spec["criteria"], "criteria", 0, lines)
    for key, value in spec.items():
        if key == "criteria" or key in _NOT_RENDERED:
            continue
        _render(value, str(key), 0, lines)
    sources = []
    for key, label in (("primary_source", "primary"), ("backup_source", "backup")):
        if spec.get(key) is not None:
            sources.append(f"{label}: {_describe_source(spec[key])}")
    if sources:
        lines.append("Sources: " + "; ".join(sources))
    return "\n".join(lines)


def render_clarifications(clarifications: Sequence[Json] | None) -> str:
    out = []
    for c in clarifications or []:
        day = str(c.get("created_at") or "")[:10]
        out.append(f"Clarification ({day}): {c.get('text', '')}")
    return "\n".join(out)


def page_url(q: Json, api_url: str = PRODUCTION_API_URL) -> str:
    """The question's public page (``/q/{slug}`` on the API host, as ``apps/api/src/pages``
    builds it), or the API URL ``/v1/questions/{id}`` in the practice sandbox, which has no
    question pages."""
    base = api_url.rstrip("/")
    if q.get("practice") or not q.get("slug"):
        return f"{base}/v1/questions/{urllib.parse.quote(str(q['id']), safe='')}"
    return f"{base}/q/{urllib.parse.quote(str(q['slug']), safe='')}"


def _sorted_bins(q: Json) -> list[Json]:
    bins = sorted(q.get("bins") or [], key=lambda b: b["idx"])
    if not bins:
        raise MappingError(f"numeric question {q.get('id')} has no bins")
    for i, b in enumerate(bins):
        if b["lower"] is None and i != 0:
            raise MappingError(f"bin {b['idx']} has an open lower edge but is not the first")
        if b["upper"] is None and i != len(bins) - 1:
            raise MappingError(f"bin {b['idx']} has an open upper edge but is not the last")
        if i > 0 and b["lower"] != bins[i - 1]["upper"]:
            raise MappingError(f"bins {bins[i - 1]['idx']} and {b['idx']} are not contiguous")
        if b["lower"] is not None and b["upper"] is not None and b["lower"] >= b["upper"]:
            raise MappingError(f"bin {b['idx']} is empty")
    return bins


def _finite_edges(bins: list[Json]) -> list[float]:
    edges = {float(e) for b in bins for e in (b["lower"], b["upper"]) if e is not None}
    return sorted(edges)


def presented_as(q: Json) -> str:
    """How the question is shown to the bot: ``binary``, ``multiple_choice`` or ``numeric``. A
    numeric question with fewer than two distinct finite bin edges (a single threshold) has no
    meaningful range and is shown as multiple choice over its bins."""
    fmt = q.get("format")
    if fmt == "numeric":
        return "numeric" if len(_finite_edges(_sorted_bins(q))) >= 2 else "multiple_choice"
    if fmt in ("binary", "multiple_choice"):
        return fmt
    raise MappingError(f"question {q.get('id')} has unsupported format {fmt!r}")


def _option_labels(q: Json) -> list[str]:
    if q.get("format") == "numeric":
        return [describe_bin(b) for b in _sorted_bins(q)]
    options = sorted(q.get("options") or [], key=lambda o: o["idx"])
    labels = [str(o["label"]) for o in options]
    if len(labels) < 2:
        raise MappingError(f"question {q.get('id')} has fewer than two options")
    if len(set(labels)) != len(labels):
        raise MappingError(f"question {q.get('id')} has duplicate option labels")
    return labels


def to_forecasting_tools_question(
    q: Json, api_url: str = PRODUCTION_API_URL
) -> BinaryQuestion | MultipleChoiceQuestion | NumericQuestion:
    """Map a full Arctend question (``GET /v1/questions/{id}``) to a ``forecasting_tools``
    question. Pure: the same input always gives the same output (``date_accessed`` aside, which
    the package sets itself). ``api_url`` is the deployment the question came from, used for
    ``page_url``."""
    shape = presented_as(q)
    api_json = dict(q)
    if shape == "multiple_choice" and q.get("format") == "numeric":
        api_json[PRESENTED_AS_KEY] = "multiple_choice"
    spec = q.get("resolution_spec") or {}
    unit = spec.get("unit") if isinstance(spec, dict) else None
    common: Json = {
        "question_text": q["title"],
        "background_info": q.get("body") or "",
        "resolution_criteria": render_resolution_criteria(spec),
        "fine_print": render_clarifications(q.get("clarifications")),
        "open_time": _parse_time(q.get("open_at")),
        "close_time": _parse_time(q.get("close_at")),
        "scheduled_resolution_time": _parse_time(q.get("resolve_at")),
        "page_url": page_url(q, api_url),
        "id_of_post": None,
        "id_of_question": None,
        "tournament_slugs": ["arctend", "arctend-sandbox"] if q.get("practice") else ["arctend"],
        "api_json": api_json,
        "unit_of_measure": str(unit) if unit is not None else None,
    }
    if shape == "binary":
        return BinaryQuestion(**common, community_prediction_at_access_time=None)
    if shape == "multiple_choice":
        return MultipleChoiceQuestion(**common, options=_option_labels(q))
    bins = _sorted_bins(q)
    edges = _finite_edges(bins)
    return NumericQuestion(
        **common,
        lower_bound=edges[0],
        upper_bound=edges[-1],
        open_lower_bound=bins[0]["lower"] is None,
        open_upper_bound=bins[-1]["upper"] is None,
        zero_point=None,
        cdf_size=CDF_SIZE,
    )


# --- forecasting_tools prediction -> Arctend submission -------------------------------------


def _utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def _cut_utf16(text: str, limit: int) -> str:
    out = []
    used = 0
    for ch in text:
        size = 2 if ord(ch) > 0xFFFF else 1
        if used + size > limit:
            break
        out.append(ch)
        used += size
    return "".join(out)


def cut_rationale(text: str, limit: int = RATIONALE_MAX) -> str:
    """Cut ``text`` to ``limit`` UTF-16 code units, at the last sentence boundary when one falls
    in the second half; otherwise at the limit."""
    text = text.strip()
    if _utf16_len(text) <= limit:
        return text
    head = _cut_utf16(text, limit)
    best = -1
    for mark in (". ", "! ", "? ", ".\n", "!\n", "?\n", "\n\n"):
        pos = head.rfind(mark)
        if pos > best:
            best = pos
    if best >= len(head) // 2:
        return head[: best + 1].rstrip()
    return head.rstrip()


def report_reasoning(report: Any) -> str | None:
    """The forecasters' reasoning from a report (the ``# FORECASTS`` section that
    ``ForecastBot`` writes), without its top heading. Never the summary or research sections,
    which may contain copied source material. None when the section cannot be found (for
    example, a bot that writes its own explanation format)."""
    try:
        section = report.forecast_rationales
    except Exception:  # noqa: BLE001 - any failure means: no rationale rather than research text
        return None
    lines = str(section).strip().splitlines()
    if lines and lines[0].lstrip().startswith("#") and "forecast" in lines[0].lower():
        lines = lines[1:]
    text = "\n".join(lines).strip()
    return text or None


def default_agent_version(bot_class_name: str = "ForecastBot") -> str:
    return f"{bot_class_name} via arctend-python/{VERSION}"[:AGENT_VERSION_MAX]


def _finite_probability(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise MappingError(f"{what} is not a number: {value!r}")
    p = float(value)
    if not math.isfinite(p) or p < 0 or p > 1:
        raise MappingError(f"{what} {value!r} is outside [0, 1]")
    return p


def _normalise(vector: list[float]) -> list[float]:
    total = math.fsum(vector)
    if not math.isfinite(total) or total <= 0:
        raise MappingError("the prediction has no probability mass")
    return [x / total for x in vector]


def _vector_from_options(labels: list[str], prediction: Any) -> list[float]:
    options = getattr(prediction, "predicted_options", None)
    if options is None:
        raise MappingError(f"expected a PredictedOptionList, got {type(prediction).__name__}")
    by_name: dict[str, float] = {}
    for option in options:
        name = option.option_name
        if name in by_name:
            raise MappingError(f"option {name!r} appears twice in the prediction")
        if name not in labels:
            raise MappingError(f"unknown option {name!r} in the prediction")
        by_name[name] = _finite_probability(option.probability, f"option {name!r}")
    missing = [label for label in labels if label not in by_name]
    if missing:
        raise MappingError(f"the prediction has no probability for {missing}")
    return _normalise([by_name[label] for label in labels])


def _cdf_points(prediction: Any) -> tuple[list[float], list[float]]:
    get_cdf = getattr(prediction, "get_cdf", None)
    if get_cdf is None:
        raise MappingError(f"expected a NumericDistribution, got {type(prediction).__name__}")
    try:
        cdf = get_cdf()
    except Exception as err:  # noqa: BLE001 - the package's own validation failed
        raise MappingError(f"the numeric distribution is invalid: {err}") from err
    points = sorted((float(p.value), float(p.percentile)) for p in cdf)
    if len(points) < 2:
        raise MappingError("the numeric distribution has fewer than two CDF points")
    values = [v for v, _ in points]
    heights = [h for _, h in points]
    if any(not math.isfinite(x) for x in values + heights):
        raise MappingError("the numeric distribution has a non-finite CDF point")
    return values, heights


def _cdf_at(values: list[float], heights: list[float], x: float) -> float:
    """The CDF at ``x``, linearly interpolated between points; flat outside the range."""
    if x <= values[0]:
        return heights[0]
    if x >= values[-1]:
        return heights[-1]
    i = bisect_right(values, x)
    x0, x1 = values[i - 1], values[i]
    y0, y1 = heights[i - 1], heights[i]
    if x1 == x0:
        return y1
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _vector_from_cdf(bins: list[Json], prediction: Any) -> list[float]:
    """Bin mass = CDF(upper edge) - CDF(lower edge). The first bin starts at CDF 0 and the last
    ends at CDF 1, so an open tail gets the CDF value at ``lower_bound`` (or one minus the value
    at ``upper_bound``) and a closed outer bin absorbs any mass the package leaves outside the
    range."""
    values, heights = _cdf_points(prediction)
    last = len(bins) - 1
    masses = []
    for i, b in enumerate(bins):
        lo = 0.0 if i == 0 else _cdf_at(values, heights, float(b["lower"]))
        hi = 1.0 if i == last else _cdf_at(values, heights, float(b["upper"]))
        masses.append(max(0.0, hi - lo))
    return _normalise(masses)


def to_arctend_submission(
    q: Json,
    report: ForecastReport | Any,
    *,
    include_rationale: bool = True,
    agent_version: str | None = None,
) -> Json:
    """Map a ``ForecastReport`` for question ``q`` to the body of
    ``PUT /v1/questions/{id}/forecast``. Raises ``MappingError`` when the prediction does not fit
    the question."""
    shape = presented_as(q)
    prediction = report.prediction
    body: Json
    if shape == "binary":
        body = {"probability": _finite_probability(prediction, "binary prediction")}
    elif shape == "multiple_choice":
        body = {"vector": _vector_from_options(_option_labels(q), prediction)}
    else:
        body = {"vector": _vector_from_cdf(_sorted_bins(q), prediction)}
    if include_rationale:
        reasoning = report_reasoning(report)
        if reasoning:
            body["rationale"] = cut_rationale(reasoning)
    version = (agent_version or default_agent_version()).strip()
    body["agent_version"] = _cut_utf16(version, AGENT_VERSION_MAX) or default_agent_version()
    return body
