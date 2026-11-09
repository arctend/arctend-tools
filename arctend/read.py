"""The read commands of ``arctend``: plain-text rendering of API bodies.

Standard library only, and never imports ``forecasting_tools``. Every function returns text; the
command line prints it (or the raw JSON with ``--json``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

Json = dict[str, Any]


def _scalar(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def key_values(value: Any, indent: int = 0) -> list[str]:
    """``key: value`` lines; nested objects indented, lists of scalars on one line."""
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, dict):
                lines.append(f"{pad}{key}:")
                lines.extend(key_values(item, indent + 1))
            elif isinstance(item, list):
                if all(not isinstance(v, dict | list) for v in item):
                    lines.append(f"{pad}{key}: {', '.join(_scalar(v) for v in item) or '-'}")
                else:
                    lines.append(f"{pad}{key}:")
                    for sub in item:
                        lines.append(f"{pad}  -")
                        lines.extend(key_values(sub, indent + 2))
            else:
                lines.append(f"{pad}{key}: {_scalar(item)}")
    else:
        lines.append(f"{pad}{_scalar(value)}")
    return lines


def utc(value: str | None) -> str:
    """An ISO time as ``YYYY-MM-DD HH:MM UTC`` (the API sends UTC)."""
    if not value:
        return "-"
    text = str(value)
    return f"{text[:10]} {text[11:16]} UTC" if len(text) >= 16 else text


def questions_table(items: Sequence[Json]) -> str:
    if not items:
        return "No open questions."
    return "\n".join(
        f"{q.get('id')}  {str(q.get('format')):<15}  {utc(q.get('close_at'))}  {q.get('title')}"
        for q in items
    )


def question_text(q: Json) -> str:
    lines = [
        f"title: {q.get('title')}",
        f"id: {q.get('id')}",
        f"format: {q.get('format')}",
        f"status: {q.get('status')}",
        f"topic: {q.get('topic_primary')}",
        f"opens: {utc(q.get('open_at'))}",
        f"closes: {utc(q.get('close_at'))}",
        f"resolves: {utc(q.get('resolve_at'))}",
    ]
    if q.get("final_outcome") is not None:
        lines.append(f"final outcome: {_scalar(q['final_outcome'])}")
    for o in sorted(q.get("options") or [], key=lambda o: o["idx"]):
        lines.append(f"option {o['idx']}: {o.get('label')}")
    for b in sorted(q.get("bins") or [], key=lambda b: b["idx"]):
        lines.append(f"bin {b['idx']}: {bin_label(b)}")
    if q.get("body"):
        lines += ["", str(q["body"])]
    spec = q.get("resolution_spec")
    if spec:
        lines += ["", "resolution:"]
        lines += key_values(spec, 1)
    for c in q.get("clarifications") or []:
        lines.append(f"clarification ({str(c.get('created_at') or '')[:10]}): {c.get('text', '')}")
    return "\n".join(lines)


def _number(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else repr(float(x))


def bin_label(b: Json) -> str:
    lower = "(-infinity" if b.get("lower") is None else f"[{_number(b['lower'])}"
    upper = "+infinity)" if b.get("upper") is None else f"{_number(b['upper'])})"
    return f"{lower}, {upper}"


def _labels(question: Json | None, fmt: str, n: int) -> list[str]:
    if fmt == "binary":
        return ["yes", "no"][:n]
    if question is not None:
        if fmt == "numeric" and question.get("bins"):
            return [bin_label(b) for b in sorted(question["bins"], key=lambda b: b["idx"])]
        if fmt == "multiple_choice" and question.get("options"):
            opts = sorted(question["options"], key=lambda o: o["idx"])
            return [str(o.get("label")) for o in opts]
    return [str(i) for i in range(n)]


def _point_header(p: Json) -> list[str]:
    return [
        f"n_forecasters: {_scalar(p.get('n_forecasters'))}",
        f"n_attested: {_scalar(p.get('n_attested'))}",
        f"thin_crowd: {_scalar(p.get('thin_crowd'))}",
        f"computed_at: {p.get('computed_at')}",
    ]


def _vector_lines(value: Sequence[float], labels: Sequence[str], fmt: str) -> list[str]:
    if fmt == "binary":
        return [f"probability: {value[0]:.4f}"]
    return [f"{label}: {p:.4f}" for label, p in zip(labels, value, strict=False)]


def consensus_text(body: Json, question: Json | None = None) -> str:
    fmt = str(body.get("format"))
    median = body.get("median")
    if not median:
        return f"No consensus yet (as of {body.get('as_of')})."
    lines = [f"question: {body.get('question_id')}", f"format: {fmt}", "median:"]
    labels = _labels(question, fmt, len(median["value"]))
    lines += ["  " + ln for ln in _vector_lines(median["value"], labels, fmt)]
    lines += _point_header(median)
    return "\n".join(lines)


def history_text(body: Json) -> str:
    items = body.get("items") or []
    if not items:
        return "No consensus history yet."
    out = []
    for p in items:
        value = ", ".join(f"{v:.4f}" for v in p.get("value") or [])
        out.append(
            f"{p.get('computed_at')}  [{value}]  n_forecasters {p.get('n_forecasters')}  "
            f"n_attested {p.get('n_attested')}  thin_crowd {_scalar(p.get('thin_crowd'))}"
        )
    if body.get("next_cursor"):
        out.append("(more points exist; this is the newest page)")
    return "\n".join(out)


def _score(value: Any) -> str:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return f"{value:.4f}"
    return _scalar(value)


def results_table(items: Sequence[Json]) -> str:
    """Practice results, one line each: received time (UTC), format, log score, the outcome
    index and the question title."""
    return "\n".join(
        f"{utc(r.get('forecast_received_at'))}  {str(r.get('format')):<15}  "
        f"log_score {_score(r.get('log_score'))}  outcome {_scalar(r.get('outcome_index'))}  "
        f"{r.get('question_title')}"
        for r in items
    )


def key_value_text(value: Any) -> str:
    return "\n".join(key_values(value))
