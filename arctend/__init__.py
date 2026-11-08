"""Arctend adapter for bots built on ``forecasting-tools`` (Metaculus's bot template).

Register with Arctend, pull Arctend's open questions in the classes the bot already consumes, and
post the bot's predictions back. The bot's research and forecasting code does not change.

``ArctendClient`` and the registration helpers are standard library only. The mapping and the
runner import ``forecasting_tools`` (slow to import, and it may read model-provider settings), so
they load on first use: ``check`` and ``register`` never import it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .api import VERSION, ApiError, ArctendClient, RedirectRefusedError

if TYPE_CHECKING:
    from .mapping import MappingError, to_arctend_submission, to_forecasting_tools_question
    from .runner import ArctendRunResult, ArctendStopError, forecast_on_arctend

__version__ = VERSION

_LAZY = {
    "forecast_on_arctend": "runner",
    "ArctendRunResult": "runner",
    "ArctendStopError": "runner",
    "to_forecasting_tools_question": "mapping",
    "to_arctend_submission": "mapping",
    "MappingError": "mapping",
}

__all__ = [
    "VERSION",
    "ApiError",
    "ArctendClient",
    "ArctendRunResult",
    "ArctendStopError",
    "MappingError",
    "RedirectRefusedError",
    "forecast_on_arctend",
    "to_arctend_submission",
    "to_forecasting_tools_question",
]


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'arctend' has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module}", __name__), name)
