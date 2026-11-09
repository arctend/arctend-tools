"""Run a Metaculus bot-template bot (``forecasting-tools``) on Arctend's questions.

Install with ``pip install "arctend[tools]"``. Importing this package imports
``forecasting_tools`` (slow, and it may read model-provider settings); ``import arctend`` and
``python -m arctend register | check`` never do.
"""

from __future__ import annotations

from .mapping import MappingError, to_arctend_submission, to_forecasting_tools_question
from .runner import ArctendRunResult, ArctendStopError, forecast_on_arctend, summarise

__all__ = [
    "ArctendRunResult",
    "ArctendStopError",
    "MappingError",
    "forecast_on_arctend",
    "summarise",
    "to_arctend_submission",
    "to_forecasting_tools_question",
]
