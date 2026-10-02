from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

_log = logging.getLogger(__name__)


def post(url: str | None, text: str, *, timeout_s: float = 10.0) -> bool:
    """POST a Slack-compatible `{"text": ...}` body to `url`.

    Network errors are logged and swallowed — a webhook outage must never break
    the pipeline. Returns True on a 2xx response, False otherwise.
    """
    if not url:
        return False
    payload = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            resp.read()
            return 200 <= resp.status < 300
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        _log.warning("Webhook post failed: %s", exc)
        return False
