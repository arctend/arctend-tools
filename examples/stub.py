"""A forecasting-tools bot with no research and no model calls, for testing the plumbing.

    python -m arctend_forecasting_tools run --bot examples.stub:UniformBot --dry-run

It gives every outcome equal probability (numeric: spread evenly over the question's range). That
scores about zero or worse: it never beats the crowd. Use it only to check that registration,
reading questions and submitting work before you spend anything on model calls.
"""

from __future__ import annotations

from typing import Any

from forecasting_tools import (
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

REASONING = "Uniform baseline from arctend-forecasting-tools (plumbing test, no model call)."


class UniformBot(ForecastBot):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("enable_summarize_research", False)
        super().__init__(**kwargs)

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, Any]:
        # No model at all: nothing here is ever invoked.
        return {"default": None, "summarizer": None, "researcher": None, "parser": None}

    async def run_research(self, question: MetaculusQuestion) -> str:
        return ""

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        return ReasonedPrediction(prediction_value=0.5, reasoning=REASONING)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        p = 1 / len(question.options)
        options = [PredictedOption(option_name=o, probability=p) for o in question.options]
        return ReasonedPrediction(
            prediction_value=PredictedOptionList(predicted_options=options), reasoning=REASONING
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        lo, hi = question.lower_bound, question.upper_bound
        percentiles = [
            Percentile(percentile=p, value=lo + p * (hi - lo)) for p in (0.1, 0.3, 0.5, 0.7, 0.9)
        ]
        return ReasonedPrediction(
            prediction_value=NumericDistribution.from_question(percentiles, question),
            reasoning=REASONING,
        )
