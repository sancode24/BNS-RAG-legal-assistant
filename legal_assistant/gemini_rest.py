"""Gemini generateContent via REST (works on Python 3.8+ without google-generativeai)."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

from legal_assistant.config import GEMINI_API_KEY, GEMINI_MAX_RETRIES, GEMINI_MODEL


def _retry_after_seconds(error_body: str, attempt: int) -> float:
    m = re.search(r"Please retry in ([0-9.]+)s", error_body)
    if m:
        return min(45.0, float(m.group(1)) + 0.75)
    return min(30.0, 4.0 * (2**attempt))


def generate_text(prompt: str, temperature: float = 0.2) -> str:
    if not GEMINI_API_KEY:
        return "Set GEMINI_API_KEY in a `.env` file in the project root (see `.env.example`)."

    url = (
        "https://generativelanguage.googleapis.com/v1/"
        f"models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    body: dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 2048,
        },
    }
    data = json.dumps(body).encode("utf-8")

    for attempt in range(GEMINI_MAX_RETRIES):
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            if e.code == 429 and attempt < GEMINI_MAX_RETRIES - 1:
                wait = _retry_after_seconds(err, attempt)
                time.sleep(wait)
                continue
            return f"Gemini API error ({e.code}): {err[:2000]}"
        except Exception as e:
            return f"Request failed: {e!s}"

        try:
            parts = payload["candidates"][0]["content"]["parts"]
            texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
            out = "".join(texts).strip()
            return out or "Empty model response."
        except (KeyError, IndexError, TypeError):
            return f"Unexpected API response: {json.dumps(payload)[:1500]}"
