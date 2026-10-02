from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from deciphaer_image_segmentation import notify


def test_post_with_no_url_returns_false() -> None:
    assert notify.post(None, "hello") is False
    assert notify.post("", "hello") is False


def test_post_success_sends_slack_payload() -> None:
    resp = MagicMock()
    resp.status = 200
    resp.read.return_value = b""
    cm = MagicMock()
    cm.__enter__.return_value = resp
    cm.__exit__.return_value = False

    with patch("urllib.request.urlopen", return_value=cm) as urlopen:
        ok = notify.post("https://example.invalid/webhook", "hi there")
    assert ok is True
    # Verify the body is a JSON {"text": ...} payload.
    req = urlopen.call_args.args[0]
    body = json.loads(req.data.decode("utf-8"))
    assert body == {"text": "hi there"}
    assert req.headers.get("Content-type") == "application/json"


def test_post_network_failure_returns_false_and_does_not_raise() -> None:
    import urllib.error

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("boom")):
        assert notify.post("https://example.invalid/webhook", "x") is False


def test_post_non_2xx_status_returns_false() -> None:
    resp = MagicMock()
    resp.status = 500
    resp.read.return_value = b""
    cm = MagicMock()
    cm.__enter__.return_value = resp
    cm.__exit__.return_value = False
    with patch("urllib.request.urlopen", return_value=cm):
        assert notify.post("https://example.invalid/webhook", "x") is False
