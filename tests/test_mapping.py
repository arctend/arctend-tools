"""Acceptance tests 1 to 4: the mapping both ways, from what forecasting-tools actually returns."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import pytest
from conftest import make_question, two_bins, uniform_distribution
from forecasting_tools import (
    BinaryQuestion,
    BinaryReport,
    MultipleChoiceQuestion,
    MultipleChoiceReport,
    NumericDistribution,
    NumericQuestion,
    NumericReport,
    Percentile,
    PredictedOption,
    PredictedOptionList,
)

from arctend.api import VERSION
from arctend.mapping import (
    MappingError,
    cut_rationale,
    describe_bin,
    page_url,
    render_resolution_criteria,
    to_arctend_submission,
    to_forecasting_tools_question,
)

RESEARCH_SENTINEL = "COPIED-SOURCE-MATERIAL-FROM-RESEARCH"


def explanation(reasoning: str = "The base rate is about 60 percent.") -> str:
    return (
        "# SUMMARY\n*Question*: q\n\n## Report 1 Summary\n### Research Summary\n"
        f"{RESEARCH_SENTINEL} summary\n\n# RESEARCH\n## Report 1 Research\n{RESEARCH_SENTINEL}\n\n"
        f"# FORECASTS\n\n## R1: Forecaster 1 Reasoning\n{reasoning}\n"
    )


# --- 1. Binary ----------------------------------------------------------------------------------


def test_binary_question_fields() -> None:
    q = make_question("binary", 1)
    ft = to_forecasting_tools_question(q, "https://api.arctend.com")
    assert isinstance(ft, BinaryQuestion)
    assert ft.question_text == q["title"]
    assert ft.background_info == q["body"]
    assert ft.community_prediction_at_access_time is None
    assert ft.api_json == q
    assert ft.tournament_slugs == ["arctend"]
    assert ft.id_of_post is None and ft.id_of_question is None
    assert ft.already_forecasted is None
    assert ft.unit_of_measure == "percent"
    assert ft.page_url == "https://api.arctend.com/q/question-1"
    for field, key in (
        ("open_time", "open_at"),
        ("close_time", "close_at"),
        ("scheduled_resolution_time", "resolve_at"),
    ):
        value = getattr(ft, field)
        assert value.tzinfo is not None and value.utcoffset().total_seconds() == 0
        assert value == datetime.fromisoformat(q[key].replace("Z", "+00:00")).astimezone(UTC)


def test_practice_question_is_tagged_and_links_to_the_api() -> None:
    q = make_question("binary", 2, practice=True)
    ft = to_forecasting_tools_question(q, "https://sandbox.arctend.com/")
    assert ft.tournament_slugs == ["arctend", "arctend-sandbox"]
    assert ft.page_url == f"https://sandbox.arctend.com/v1/questions/{q['id']}"
    assert page_url(make_question("binary", 3, slug="a b/c")) == (
        "https://api.arctend.com/q/a%20b%2Fc"
    )


def test_binary_report_maps_to_probability() -> None:
    q = make_question("binary", 1)
    ft = to_forecasting_tools_question(q)
    report = BinaryReport(question=ft, prediction=0.62, explanation=explanation())
    body = to_arctend_submission(q, report, agent_version="MyBot via test")
    assert body == {
        "probability": 0.62,
        "rationale": "## R1: Forecaster 1 Reasoning\nThe base rate is about 60 percent.",
        "agent_version": "MyBot via test",
    }
    assert RESEARCH_SENTINEL not in body["rationale"]
    no_rationale = to_arctend_submission(q, report, include_rationale=False)
    assert "rationale" not in no_rationale
    assert no_rationale["agent_version"] == f"ForecastBot via arctend-python/{VERSION}"


@pytest.mark.parametrize("bad", [1.2, -0.1, float("nan"), "0.5", True])
def test_binary_out_of_range_is_an_error(bad: Any) -> None:
    q = make_question("binary", 1)
    ft = to_forecasting_tools_question(q)
    report = BinaryReport.model_construct(question=ft, prediction=bad, explanation=explanation())
    with pytest.raises(MappingError):
        to_arctend_submission(q, report)


def test_custom_explanation_sends_no_rationale() -> None:
    q = make_question("binary", 1)
    ft = to_forecasting_tools_question(q)
    report = BinaryReport(
        question=ft, prediction=0.3, explanation=f"# My notes\n{RESEARCH_SENTINEL}"
    )
    assert "rationale" not in to_arctend_submission(q, report)


def test_agent_version_is_cut_to_64() -> None:
    q = make_question("binary", 1)
    report = BinaryReport(
        question=to_forecasting_tools_question(q), prediction=0.5, explanation=explanation()
    )
    body = to_arctend_submission(q, report, agent_version="x" * 100)
    assert body["agent_version"] == "x" * 64


# --- 2. Multiple choice -------------------------------------------------------------------------


def mc_report(q: dict, ft: Any, pairs: list[tuple[str, float]]) -> Any:
    options = [PredictedOption.model_construct(option_name=n, probability=p) for n, p in pairs]
    return MultipleChoiceReport.model_construct(
        question=ft,
        prediction=PredictedOptionList.model_construct(predicted_options=options),
        explanation=explanation(),
    )


def test_multiple_choice_options_in_idx_order() -> None:
    q = make_question("multiple_choice", 4)  # idx 2, 0, 1 in the fixture
    ft = to_forecasting_tools_question(q)
    assert isinstance(ft, MultipleChoiceQuestion)
    assert ft.options == ["Red", "Blue", "Green"]
    assert ft.api_json == q
    pred = PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name="Green", probability=0.2),
            PredictedOption(option_name="Red", probability=0.5),
            PredictedOption(option_name="Blue", probability=0.3),
        ]
    )
    report = MultipleChoiceReport(question=ft, prediction=pred, explanation=explanation())
    body = to_arctend_submission(q, report)
    assert body["vector"] == pytest.approx([0.5, 0.3, 0.2], abs=1e-9)
    assert math.fsum(body["vector"]) == pytest.approx(1, abs=1e-9)


def test_multiple_choice_is_renormalised() -> None:
    q = make_question("multiple_choice", 4)
    ft = to_forecasting_tools_question(q)
    pairs = [("Red", 0.25), ("Blue", 0.25), ("Green", 0.505)]  # sums to 1.005
    body = to_arctend_submission(q, mc_report(q, ft, pairs))
    assert body["vector"] == pytest.approx([0.25 / 1.005, 0.25 / 1.005, 0.505 / 1.005])
    assert math.fsum(body["vector"]) == pytest.approx(1, abs=1e-9)


@pytest.mark.parametrize(
    "pairs",
    [
        [("Red", 0.5), ("Blue", 0.5)],  # Green missing
        [("Red", 0.4), ("Blue", 0.3), ("Green", 0.2), ("Purple", 0.1)],  # unknown option
        [("Red", 0.4), ("Red", 0.3), ("Green", 0.3)],  # duplicate name
    ],
)
def test_multiple_choice_missing_or_unknown_option_is_an_error(pairs: list) -> None:
    q = make_question("multiple_choice", 4)
    ft = to_forecasting_tools_question(q)
    with pytest.raises(MappingError):
        to_arctend_submission(q, mc_report(q, ft, pairs))


def test_duplicate_labels_are_rejected() -> None:
    q = make_question(
        "multiple_choice", 5, options=[{"idx": 0, "label": "A"}, {"idx": 1, "label": "A"}]
    )
    with pytest.raises(MappingError, match="duplicate"):
        to_forecasting_tools_question(q)


# --- 3. Numeric ---------------------------------------------------------------------------------


def test_cdf_convention_of_the_installed_package() -> None:
    """The convention the bin masses rely on (spec 1.3), read from the package itself."""
    q = make_question("numeric", 6)
    ft = to_forecasting_tools_question(q)
    dist = NumericDistribution.from_question(
        [Percentile(value=v, percentile=p) for v, p in ((6, 0.1), (30, 0.5), (54, 0.9))], ft
    )
    cdf = dist.get_cdf()
    assert len(cdf) == 201
    assert cdf[0].value == ft.lower_bound and cdf[-1].value == ft.upper_bound
    assert all(0 <= p.percentile <= 1 for p in cdf)  # a 0 to 1 scale, not 0 to 100
    assert cdf[0].percentile > 0 and cdf[-1].percentile < 1  # open tails carry mass


def test_numeric_bounds_and_open_flags() -> None:
    q = make_question("numeric", 6)
    ft = to_forecasting_tools_question(q)
    assert isinstance(ft, NumericQuestion)
    assert (ft.lower_bound, ft.upper_bound) == (0.0, 60.0)
    assert ft.open_lower_bound is True and ft.open_upper_bound is True
    assert ft.zero_point is None and ft.cdf_size == 201
    closed = make_question(
        "numeric",
        7,
        bins=[
            {"idx": 0, "lower": 0, "upper": 5, "lower_open": False, "upper_open": False},
            {"idx": 1, "lower": 5, "upper": 9, "lower_open": False, "upper_open": False},
        ],
    )
    ft_closed = to_forecasting_tools_question(closed)
    assert (ft_closed.lower_bound, ft_closed.upper_bound) == (0.0, 9.0)
    assert ft_closed.open_lower_bound is False and ft_closed.open_upper_bound is False


def test_uniform_over_the_range_maps_to_widths() -> None:
    q = make_question("numeric", 6)
    ft = to_forecasting_tools_question(q)
    dist = uniform_distribution(ft)
    cdf = dist.get_cdf()
    assert cdf[0].percentile == pytest.approx(0) and cdf[-1].percentile == pytest.approx(1)
    report = NumericReport(question=ft, prediction=dist, explanation=explanation())
    vector = to_arctend_submission(q, report)["vector"]
    assert vector == pytest.approx([0, 10 / 60, 20 / 60, 30 / 60, 0], abs=1e-6)
    assert math.fsum(vector) == pytest.approx(1, abs=1e-9)
    assert all(x >= 0 for x in vector)


def test_declared_5th_percentile_below_lower_bound_fills_the_lower_tail() -> None:
    q = make_question("numeric", 6)
    ft = to_forecasting_tools_question(q)
    points = [Percentile(value=v, percentile=p) for v, p in ((-15, 0.05), (30, 0.5), (54, 0.9))]
    exact = NumericDistribution.from_question(points, ft, standardize_cdf=False)
    vector = to_arctend_submission(
        q, NumericReport.model_construct(question=ft, prediction=exact, explanation=explanation())
    )["vector"]
    # The CDF interpolates linearly from (-15, 0.05) to (30, 0.5): at 0 it is 0.05 + 15/45 * 0.45.
    assert vector[0] == pytest.approx(0.2, abs=1e-6)
    assert math.fsum(vector) == pytest.approx(1, abs=1e-9)

    standardised = NumericDistribution.from_question(points, ft)
    cdf = standardised.get_cdf()
    vector2 = to_arctend_submission(
        q,
        NumericReport.model_construct(
            question=ft, prediction=standardised, explanation=explanation()
        ),
    )["vector"]
    assert vector2[0] == pytest.approx(cdf[0].percentile, abs=1e-9)
    assert vector2[-1] == pytest.approx(1 - cdf[-1].percentile, abs=1e-9)
    assert vector2[0] > 0.05


def test_closed_outer_bins_absorb_mass_outside_the_range() -> None:
    q = make_question(
        "numeric",
        8,
        bins=[
            {"idx": 0, "lower": 0, "upper": 5, "lower_open": False, "upper_open": False},
            {"idx": 1, "lower": 5, "upper": 10, "lower_open": False, "upper_open": False},
        ],
    )
    ft = to_forecasting_tools_question(q)
    # A distribution built as if the bounds were open leaves mass outside [0, 10].
    points = [Percentile(value=v, percentile=p) for v, p in ((1, 0.1), (5, 0.5), (9, 0.9))]
    dist = NumericDistribution(
        declared_percentiles=points,
        open_lower_bound=True,
        open_upper_bound=True,
        lower_bound=0,
        upper_bound=10,
        zero_point=None,
        cdf_size=201,
        standardize_cdf=False,
    )
    vector = to_arctend_submission(
        q, NumericReport.model_construct(question=ft, prediction=dist, explanation=explanation())
    )["vector"]
    assert vector == pytest.approx([0.5, 0.5], abs=1e-9)


def test_single_threshold_is_presented_as_multiple_choice() -> None:
    q = make_question("numeric", 9, bins=two_bins())
    ft = to_forecasting_tools_question(q)
    assert isinstance(ft, MultipleChoiceQuestion)
    assert ft.options == ["(-infinity, 50)", "[50, +infinity)"]
    assert ft.options == [describe_bin(b) for b in two_bins()]
    assert ft.api_json["arctend_presented_as"] == "multiple_choice"
    assert "arctend_presented_as" not in q  # the input is not changed
    pred = PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name="[50, +infinity)", probability=0.7),
            PredictedOption(option_name="(-infinity, 50)", probability=0.3),
        ]
    )
    report = MultipleChoiceReport(question=ft, prediction=pred, explanation=explanation())
    assert to_arctend_submission(q, report)["vector"] == pytest.approx([0.3, 0.7])


def test_non_contiguous_bins_are_rejected() -> None:
    bins = two_bins()
    bins[1]["lower"] = 60.0
    with pytest.raises(MappingError, match="contiguous"):
        to_forecasting_tools_question(make_question("numeric", 10, bins=bins))


def test_wrong_prediction_type_is_an_error() -> None:
    q = make_question("numeric", 6)
    ft = to_forecasting_tools_question(q)
    report = NumericReport.model_construct(question=ft, prediction=0.5, explanation=explanation())
    with pytest.raises(MappingError):
        to_arctend_submission(q, report)


# --- 4. Resolution text and clarifications ------------------------------------------------------


def test_resolution_criteria_is_readable_lines() -> None:
    q = make_question("binary", 11)
    text = to_forecasting_tools_question(q).resolution_criteria
    assert "{" not in text and "}" not in text and '"' not in text
    lines = text.splitlines()
    assert lines[0] == "criteria: YES if the published rate is above 4."
    assert "edge cases:" in lines
    assert "  postponed: Waits for the publication" in lines
    assert "  threshold: 4" in lines
    assert lines[-1] == (
        "Sources: primary: Central bank (https://example.org/rates); "
        "backup: Statistics office (https://example.net/stats)"
    )


def test_practice_answer_key_is_never_rendered() -> None:
    spec = {"criteria": "c", "true_probabilities": [0.25, 0.75], "list": [{"a": 1}, 2]}
    text = render_resolution_criteria(spec)
    assert "true" not in text and "0.75" not in text
    assert "list:\n  -:\n    a: 1\n  - 2" in text


def test_clarifications_in_order_with_dates() -> None:
    q = make_question(
        "binary",
        12,
        clarifications=[
            {
                "id": "a",
                "version": 1,
                "text": "First note.",
                "created_by": "x",
                "created_at": "2026-10-01T03:00:00.000Z",
                "content_hash": "0" * 64,
            },
            {
                "id": "b",
                "version": 2,
                "text": "Second note.",
                "created_by": "x",
                "created_at": "2026-10-09T23:59:00.000Z",
                "content_hash": "1" * 64,
            },
        ],
    )
    ft = to_forecasting_tools_question(q)
    assert ft.fine_print == (
        "Clarification (2026-10-01): First note.\nClarification (2026-10-09): Second note."
    )
    assert to_forecasting_tools_question(make_question("binary", 13)).fine_print == ""


# --- Rationale cut ------------------------------------------------------------------------------


def test_rationale_cut_at_a_sentence_boundary() -> None:
    text = " ".join(f"Sentence number {i} explains a point." for i in range(200))
    cut = cut_rationale(text)
    assert len(cut) <= 2000 and cut.endswith(".")
    assert text.startswith(cut)


def test_rationale_limit_counts_utf16_units() -> None:
    text = "\U0001f600" * 1500  # 3,000 UTF-16 code units, no sentence boundary
    cut = cut_rationale(text)
    assert len(cut.encode("utf-16-le")) // 2 <= 2000
    assert len(cut) == 1000
