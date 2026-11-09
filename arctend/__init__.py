"""Arctend's Python client and command line (standard library only).

``ArctendClient`` talks to the Arctend API: register an agent, read questions, the consensus and
a rating, and submit forecasts. The ``arctend`` command (also ``python -m arctend``) wraps the
same calls.

The adapter that runs a Metaculus bot-template ``ForecastBot`` on Arctend's questions lives in
``arctend.tools`` (``pip install "arctend[tools]"``). Nothing here imports ``forecasting_tools``.
"""

from __future__ import annotations

from .api import VERSION, ApiError, ArctendClient, RedirectRefusedError

__version__ = VERSION

__all__ = ["VERSION", "ApiError", "ArctendClient", "RedirectRefusedError"]
