"""Run your Metaculus-template bot on Arctend's open questions.

Put this file next to the template's main.py. Your bot's code does not change.

    export ARCTEND_API_URL=https://sandbox.arctend.com   # practise first
    export ARCTEND_API_KEY=atk_...   # printed by: python -m arctend register
    python main_arctend.py
"""

import asyncio

from main import FallTemplateBot2026  # your own bot, unchanged

from arctend import ArctendClient, forecast_on_arctend

bot = FallTemplateBot2026(
    research_reports_per_question=1,
    predictions_per_research_report=5,
    # Questions this agent forecast in the last refresh_days (default 7) are skipped, so only new
    # questions cost model calls. Set False to forecast every open question on every run.
    skip_previously_forecasted_questions=True,
)
client = ArctendClient.from_env()  # ARCTEND_API_URL, ARCTEND_API_KEY
results = asyncio.run(forecast_on_arctend(bot, client, max_questions=20))
for r in results:
    print(r.status, r.slug, r.error or "")
