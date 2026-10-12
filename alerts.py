"""
Slack alerting for the trading bot.

Sends messages to a Slack channel via an "incoming webhook" — a special URL
Slack gives you that accepts simple HTTP POSTs. No Slack SDK needed; the
whole protocol is "POST a JSON body {'text': ...} to the URL".

Setup (one-time, ~10 minutes):
  1. Go to https://api.slack.com/apps -> "Create New App" -> "From scratch"
  2. Name it (e.g. "Trading Bot"), pick your workspace
  3. In the app's settings: "Incoming Webhooks" -> toggle ON
  4. "Add New Webhook to Workspace" -> choose the channel to post to
  5. Copy the webhook URL (looks like https://hooks.slack.com/services/T.../B.../x...)
  6. Add it to your .env file:
        SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

Design principles:
  - FAIL SILENT: an alerting failure must NEVER break trading. Every error
    is caught and logged; nothing propagates to the caller.
  - NO-OP WITHOUT CONFIG: if SLACK_WEBHOOK_URL isn't set, send_alert() does
    nothing (logs once). The bot runs fine without Slack configured.
  - STDLIB ONLY: uses urllib so there's no new dependency to install.
"""

import json
import logging
import os
import urllib.request

from dotenv import load_dotenv

load_dotenv()

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "").strip()

# Module-level flag so "alerting not configured" is logged once, not per call.
_warned_not_configured = False


def send_alert(text: str) -> bool:
    """Post a message to the configured Slack channel.

    Returns True if delivered, False otherwise (not configured / HTTP error).
    Never raises — see "FAIL SILENT" above.
    """
    global _warned_not_configured

    if not SLACK_WEBHOOK_URL:
        if not _warned_not_configured:
            logging.info("Slack alerting not configured (no SLACK_WEBHOOK_URL in .env). Alerts disabled.")
            _warned_not_configured = True
        return False

    try:
        payload = json.dumps({"text": text}).encode("utf-8")
        req = urllib.request.Request(
            SLACK_WEBHOOK_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # Short timeout: we'd rather drop an alert than stall the trading
        # loop for half a minute waiting on Slack.
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception as e:
        logging.error(f"Slack alert failed (trading unaffected): {e}")
        return False
