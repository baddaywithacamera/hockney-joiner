"""
vision_provider.py — Pluggable vision backends for the composition chat.

This is the abstraction the composition chat (and, later, the placement
"director") talks to. It hides whether the answer comes from a local model
or a cloud LLM (Claude / Gemini) behind one tiny interface:

    provider = get_provider()
    result = provider.query(pil_image, "Which tiles look redundant?", n_images=42)
    # result -> {"answer": str, "indices": list[int]}

Design notes
------------
* No third-party SDKs. Cloud calls use stdlib ``urllib.request`` so the cloud
  path adds **zero new hard dependencies** — it only runs when the user opts in
  and supplies a key. The offline build keeps working untouched.
* API keys come from the environment first
  (``ANTHROPIC_API_KEY`` / ``GEMINI_API_KEY``), then fall back to a value the
  user pastes into Settings (stored via QSettings). Env wins so power users /
  the author need zero in-app config and no secret is written to disk.
* The model always returns structured JSON ``{"answer", "indices"}``.
  Indices are 1-based tile numbers and are **validated against n_images**, so a
  stray number in the prose (a year, an f-stop) can never become a phantom tile
  highlight — which was a real bug in the old regex-scraping approach.

Privacy: selecting a cloud provider sends ONE contact-sheet image plus the
question to that provider. Local placement is unaffected and stays offline.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import urllib.error
import urllib.request

log = logging.getLogger(__name__)

# ── Settings keys / defaults ────────────────────────────────────────────────────

SETTINGS_ORG = "HockneyJoiner"
SETTINGS_APP = "Hockney Joiner"

KEY_PROVIDER = "vision/provider"        # "local" | "claude" | "gemini"
KEY_CLAUDE_MODEL = "vision/claude_model"
KEY_GEMINI_MODEL = "vision/gemini_model"
KEY_CLAUDE_API = "vision/claude_api_key"
KEY_GEMINI_API = "vision/gemini_api_key"

# Defaults are intentionally the cheap/fast tier — contact-sheet Q&A does not
# need a frontier model. All of these are editable in Settings because vendor
# model IDs change often; treat them as sane starting points, not gospel.
DEFAULT_PROVIDER = "local"
DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

ENV_CLAUDE_KEY = "ANTHROPIC_API_KEY"
ENV_GEMINI_KEY = "GEMINI_API_KEY"

_HTTP_TIMEOUT = 60  # seconds


# ── Config plumbing ─────────────────────────────────────────────────────────────

class ProviderConfig:
    """Resolved provider settings (provider name, model, api key)."""

    def __init__(self, provider: str, model: str, api_key: str):
        self.provider = provider
        self.model = model
        self.api_key = api_key

    @property
    def is_cloud(self) -> bool:
        return self.provider in ("claude", "gemini")

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)


def _read_settings():
    """Return a QSettings handle, or None if Qt isn't importable (tests)."""
    try:
        from PyQt6.QtCore import QSettings
    except Exception:  # pragma: no cover - Qt always present in the app
        return None
    return QSettings(SETTINGS_ORG, SETTINGS_APP)


def get_provider_config(settings=None) -> ProviderConfig:
    """
    Resolve the active provider config from QSettings, with env-var override
    for the API key (env wins). Safe to call off the GUI thread.
    """
    s = settings if settings is not None else _read_settings()

    def _get(key, default):
        if s is None:
            return default
        val = s.value(key, default)
        return val if val not in (None, "") else default

    provider = str(_get(KEY_PROVIDER, DEFAULT_PROVIDER))

    if provider == "claude":
        model = str(_get(KEY_CLAUDE_MODEL, DEFAULT_CLAUDE_MODEL))
        api_key = os.environ.get(ENV_CLAUDE_KEY) or str(_get(KEY_CLAUDE_API, ""))
    elif provider == "gemini":
        model = str(_get(KEY_GEMINI_MODEL, DEFAULT_GEMINI_MODEL))
        api_key = os.environ.get(ENV_GEMINI_KEY) or str(_get(KEY_GEMINI_API, ""))
    else:
        provider, model, api_key = "local", "", ""

    return ProviderConfig(provider, model, api_key)


# ── Shared helpers ──────────────────────────────────────────────────────────────

def _encode_jpeg(image, max_edge: int = 1568) -> str:
    """
    PIL image -> base64 JPEG string. Downscales the long edge so we don't ship a
    needlessly huge contact sheet (both APIs downsample large images anyway).
    """
    img = image
    if max(img.size) > max_edge:
        scale = max_edge / float(max(img.size))
        new_size = (max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale)))
        img = img.resize(new_size)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build_prompt(question: str, n_images: int) -> str:
    return (
        "You are a photography composition assistant for a David Hockney style "
        "'joiner' (a deliberately fragmented photo-collage). The attached image is "
        f"a numbered contact sheet of {n_images} tiles, each labelled with a "
        "1-based number in its corner.\n\n"
        f"The photographer asks: \"{question}\"\n\n"
        "Answer in 1-3 plain sentences. You advise; the photographer decides. "
        "If your answer refers to specific tiles, list their numbers.\n\n"
        "Respond with ONLY a JSON object, no markdown fences, of the form:\n"
        '{"answer": "<your 1-3 sentence reply>", "indices": [<tile numbers you referenced>]}\n'
        f"Every number in \"indices\" must be an integer between 1 and {n_images}. "
        "Use an empty list if no specific tiles apply."
    )


def _parse_json_reply(text: str, n_images: int) -> dict:
    """
    Extract {"answer", "indices"} from a model reply, tolerating stray prose or
    markdown fences. Indices are clamped to 1..n_images and de-duplicated.
    Falls back to using the raw text as the answer if no JSON is found.
    """
    answer = text.strip()
    indices: list[int] = []

    obj = None
    # Try the whole thing first, then the first {...} block.
    for candidate in (text, _first_json_block(text)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            break
        except (ValueError, TypeError):
            continue

    if isinstance(obj, dict):
        answer = str(obj.get("answer", answer)).strip() or answer
        raw = obj.get("indices", [])
        if isinstance(raw, list):
            seen = set()
            for n in raw:
                try:
                    i = int(n)
                except (ValueError, TypeError):
                    continue
                if 1 <= i <= n_images and i not in seen:
                    seen.add(i)
                    indices.append(i)

    return {"answer": answer, "indices": indices}


def _parse_json_object(text: str) -> dict:
    """Extract the first JSON *object* from a model reply, tolerating fences/prose."""
    for candidate in (text, _first_json_block(text)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    raise ProviderError("Could not parse a JSON object from the model reply.")


def _first_json_block(text: str) -> str | None:
    """Return the first balanced {...} substring, or None."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _http_post_json(url: str, payload: dict, headers: dict) -> dict:
    """POST JSON, return parsed JSON. Raises ProviderError on failure."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8")[:400]
        except Exception:
            pass
        raise ProviderError(f"HTTP {e.code} from provider: {detail or e.reason}") from e
    except urllib.error.URLError as e:
        raise ProviderError(f"Network error: {e.reason}") from e
    try:
        return json.loads(body)
    except ValueError as e:
        raise ProviderError("Provider returned non-JSON response.") from e


class ProviderError(RuntimeError):
    """Raised when a provider can't fulfil a query (network/auth/format)."""


# ── Providers ───────────────────────────────────────────────────────────────────

class VisionProvider:
    """Interface: query(image, question, n_images) -> {'answer', 'indices'}."""

    name = "base"

    def query(self, image, question: str, n_images: int) -> dict:
        raise NotImplementedError

    def query_json(self, prompt: str, images, max_tokens: int = 1500) -> dict:
        """Send a prompt + one-or-more images, return a parsed JSON object."""
        raise NotImplementedError


class ClaudeProvider(VisionProvider):
    name = "claude"
    ENDPOINT = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(self, model: str, api_key: str):
        self.model = model or DEFAULT_CLAUDE_MODEL
        self.api_key = api_key

    def _raw_complete(self, prompt: str, images, max_tokens: int) -> str:
        if not self.api_key:
            raise ProviderError(
                "No Claude API key. Set ANTHROPIC_API_KEY or add one in "
                "Help → Composition AI Settings."
            )
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": _encode_jpeg(im),
                },
            }
            for im in images
        ]
        content.append({"type": "text", "text": prompt})
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        headers = {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
        }
        return _claude_text(_http_post_json(self.ENDPOINT, payload, headers))

    def query(self, image, question: str, n_images: int) -> dict:
        text = self._raw_complete(_build_prompt(question, n_images), [image], 400)
        return _parse_json_reply(text, n_images)

    def query_json(self, prompt: str, images, max_tokens: int = 1500) -> dict:
        return _parse_json_object(self._raw_complete(prompt, images, max_tokens))


def _claude_text(resp: dict) -> str:
    blocks = resp.get("content", [])
    parts = [b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
    text = "".join(parts).strip()
    if not text:
        raise ProviderError("Claude returned an empty response.")
    return text


class GeminiProvider(VisionProvider):
    name = "gemini"
    ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, model: str, api_key: str):
        self.model = model or DEFAULT_GEMINI_MODEL
        self.api_key = api_key

    def _raw_complete(self, prompt: str, images, max_tokens: int) -> str:
        if not self.api_key:
            raise ProviderError(
                "No Gemini API key. Set GEMINI_API_KEY or add one in "
                "Help → Composition AI Settings."
            )
        parts = [
            {"inline_data": {"mime_type": "image/jpeg", "data": _encode_jpeg(im)}}
            for im in images
        ]
        parts.append({"text": prompt})
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "maxOutputTokens": max_tokens,
            },
        }
        url = self.ENDPOINT.format(model=self.model)
        headers = {
            "content-type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        return _gemini_text(_http_post_json(url, payload, headers))

    def query(self, image, question: str, n_images: int) -> dict:
        text = self._raw_complete(_build_prompt(question, n_images), [image], 600)
        return _parse_json_reply(text, n_images)

    def query_json(self, prompt: str, images, max_tokens: int = 1500) -> dict:
        return _parse_json_object(self._raw_complete(prompt, images, max_tokens))


def _gemini_text(resp: dict) -> str:
    candidates = resp.get("candidates", [])
    if not candidates:
        # Surface a safety/block reason if present.
        fb = resp.get("promptFeedback", {})
        reason = fb.get("blockReason")
        raise ProviderError(
            f"Gemini returned no answer{f' (blocked: {reason})' if reason else ''}."
        )
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
    if not text:
        raise ProviderError("Gemini returned an empty response.")
    return text


# ── Factory ─────────────────────────────────────────────────────────────────────

def get_provider(config: ProviderConfig | None = None) -> VisionProvider | None:
    """
    Build the active cloud provider, or None for the local/offline path
    (callers fall back to the on-device BLIP worker for that case).
    """
    cfg = config if config is not None else get_provider_config()
    if cfg.provider == "claude":
        return ClaudeProvider(cfg.model, cfg.api_key)
    if cfg.provider == "gemini":
        return GeminiProvider(cfg.model, cfg.api_key)
    return None


def get_placement_provider() -> VisionProvider | None:
    """
    Build a cloud provider for LLM *placement*. Prefers the configured chat
    provider; if that's local, falls back to whichever provider has a key
    available (env or settings). Returns None if no cloud key is configured.
    """
    cfg = get_provider_config()
    if cfg.is_cloud and cfg.has_key:
        return get_provider(cfg)

    s = _read_settings()

    def _key(env_name, settings_key):
        val = os.environ.get(env_name)
        if val:
            return val
        return str(s.value(settings_key, "")) if s is not None else ""

    def _model(settings_key, default):
        return str(s.value(settings_key, default)) if s is not None else default

    ck = _key(ENV_CLAUDE_KEY, KEY_CLAUDE_API)
    if ck:
        return ClaudeProvider(_model(KEY_CLAUDE_MODEL, DEFAULT_CLAUDE_MODEL), ck)
    gk = _key(ENV_GEMINI_KEY, KEY_GEMINI_API)
    if gk:
        return GeminiProvider(_model(KEY_GEMINI_MODEL, DEFAULT_GEMINI_MODEL), gk)
    return None


def provider_status(config: ProviderConfig | None = None, models_dir=None) -> tuple[bool, str]:
    """
    Is the composition chat usable right now? Returns (ok, human message).
    Used by the chat panel to decide whether to allow a question.
    """
    cfg = config if config is not None else get_provider_config()
    if cfg.is_cloud:
        if not cfg.has_key:
            return False, (
                f"{cfg.provider.title()} selected but no API key. "
                "Add one in Help → Composition AI Settings "
                f"(or set {ENV_CLAUDE_KEY if cfg.provider == 'claude' else ENV_GEMINI_KEY})."
            )
        return True, f"Using {cfg.provider.title()} ({cfg.model})."
    # Local path — defer to the BLIP readiness check the caller owns.
    if models_dir is not None:
        try:
            from hockney.core.vision_chat import is_moondream_ready
            if not is_moondream_ready(models_dir):
                return False, (
                    "Local Composition AI not downloaded. "
                    "Use Help → Download Composition AI, or switch to a "
                    "cloud provider in Help → Composition AI Settings."
                )
        except Exception:
            pass
    return True, "Using local Composition AI (offline)."


# Backwards/util: parse loose index lists out of free text, bounded by n_images.
def extract_indices(text: str, n_images: int) -> list[int]:
    """Bounded integer scrape — used only by the legacy local BLIP path."""
    out: list[int] = []
    seen = set()
    for n in re.findall(r"\b(\d+)\b", text):
        i = int(n)
        if 1 <= i <= n_images and i not in seen:
            seen.add(i)
            out.append(i)
    return out
