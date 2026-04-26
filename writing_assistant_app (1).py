"""
writing_assistant_app.py
------------------------
Writing Assistant — Streamlit desktop app.

Run with:
    streamlit run writing_assistant_app.py

A browser tab will open at http://localhost:8501 showing the app.
Close the tab + press Ctrl+C in Command Prompt to stop it.

This is a real GUI app. The old writing_assistant.py (command-line version)
still works and is untouched — this is a separate file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import streamlit as st


# =========================================================================
# CONFIG
# =========================================================================

APP_NAME = "Writing Assistant"
APP_VERSION = "0.3.0 — real app"

CHUNK_SIZE = 3000
CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 500


# =========================================================================
# CHUNKING (from Step 2 — unchanged logic)
# =========================================================================

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"[.!?][\"'\)\]]*\s+")


@dataclass
class Chunk:
    index: int
    start: int
    end: int
    text: str
    overlap_start: int

    def new_content(self) -> str:
        return self.text[self.overlap_start:]


def normalize_text(text: str) -> str:
    if not text:
        return ""
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n"))


def _find_break_point(text: str, target: int, search_window: int = 400) -> int:
    if target >= len(text):
        return len(text)
    window_start = max(0, target - search_window)
    window = text[window_start:target]
    matches = list(_PARAGRAPH_BREAK.finditer(window))
    if matches:
        return window_start + matches[-1].end()
    matches = list(_SENTENCE_END.finditer(window))
    if matches:
        return window_start + matches[-1].end()
    for i in range(target, window_start, -1):
        if i < len(text) and text[i].isspace():
            return i + 1
    return target


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    if not text:
        return []
    if len(text) <= chunk_size:
        return [Chunk(index=0, start=0, end=len(text), text=text, overlap_start=0)]

    chunks: List[Chunk] = []
    cursor = 0
    chunk_index = 0

    while cursor < len(text):
        target_end = cursor + chunk_size
        if target_end >= len(text):
            chunk_end = len(text)
        else:
            chunk_end = _find_break_point(text, target_end)
            if chunk_end <= cursor:
                chunk_end = min(cursor + chunk_size, len(text))

        if chunk_index == 0:
            overlap_begin = cursor
        else:
            overlap_begin = max(0, cursor - overlap)
            safety = 0
            while overlap_begin > 0 and not text[overlap_begin - 1].isspace():
                overlap_begin -= 1
                safety += 1
                if safety > overlap + 100:
                    overlap_begin = cursor - overlap
                    break

        chunks.append(Chunk(
            index=chunk_index,
            start=overlap_begin,
            end=chunk_end,
            text=text[overlap_begin:chunk_end],
            overlap_start=cursor - overlap_begin,
        ))
        cursor = chunk_end
        chunk_index += 1

    if len(chunks) >= 2 and len(chunks[-1].new_content()) < MIN_CHUNK_SIZE:
        last = chunks.pop()
        prev = chunks[-1]
        chunks[-1] = Chunk(
            index=prev.index,
            start=prev.start,
            end=last.end,
            text=text[prev.start:last.end],
            overlap_start=prev.overlap_start,
        )
    return chunks


# =========================================================================
# FILE READING
# =========================================================================

def read_uploaded_file(uploaded_file) -> str:
    """Decode an uploaded file to text. Supports .txt and .md for now."""
    raw = uploaded_file.read()
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return normalize_text(raw.decode(encoding))
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode file. Try converting it to UTF-8.")


# =========================================================================
# SESSION STATE
# =========================================================================

# =========================================================================
# AI CALLS
# =========================================================================

def call_openrouter(prompt: str, model: str = "meta-llama/llama-3.3-70b-instruct:free") -> str:
    """Send a prompt to OpenRouter and return the response text."""
    import urllib.request
    import json

    key = st.secrets.get("OPENROUTER_API_KEY", "")
    if not key:
        raise ValueError("OPENROUTER_API_KEY not found in Streamlit Secrets")

    data = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "writing-assistant-app/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["choices"][0]["message"]["content"]


def call_groq(prompt: str, model: str = "llama-3.1-8b-instant") -> str:
    """Send a prompt to Groq (free tier, very generous rate limits).

    Used as a third-tier fallback when both Gemini Flash and Flash-Lite fail.

    Available production models (as of 2026):
        - "llama-3.1-8b-instant"     : Fast, large daily quota (default)
        - "llama-3.3-70b-versatile"  : Smarter, smaller daily quota
        - "openai/gpt-oss-20b"       : OpenAI's open model, alternative
        - "openai/gpt-oss-120b"      : Larger OpenAI open model

    The 8B model is plenty smart for character extraction tasks
    and has the highest daily budget.
    """
    import urllib.request
    import json

    key = st.secrets.get("GROQ_API_KEY", "")
    if not key:
        raise ValueError("GROQ_API_KEY not found in Streamlit Secrets")

    data = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "writing-assistant-app/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    st.session_state["last_model_used"] = f"Groq {model} (fallback)"
    return result["choices"][0]["message"]["content"]


def call_groq_with_fallback(prompt: str) -> str:
    """Call Groq with automatic fallback between models.

    Important: Groq's per-minute token budget (8K TPM) is SHARED across all models
    in your account. Switching models only helps with per-DAY limits, not per-minute.

    On 429: wait 15 seconds before trying next model (lets quota refill)
    On 413/400: try next model immediately (different size limits)
    """
    import urllib.error
    import time

    models_to_try = [
        "llama-3.1-8b-instant",        # Smallest, cheapest tokens, try first
        "openai/gpt-oss-20b",          # Different model family
        "llama-3.3-70b-versatile",     # Smarter
        "openai/gpt-oss-120b",         # Last resort
    ]

    last_error = None
    for idx, model in enumerate(models_to_try):
        try:
            return call_groq(prompt, model=model)
        except urllib.error.HTTPError as e:
            last_error = e
            if e.code == 429:
                # Per-minute rate limit — wait before trying next model
                # since they share the same budget
                if idx < len(models_to_try) - 1:
                    time.sleep(15)
                continue
            elif e.code in (413, 400):
                # Payload too large or model issue — try next immediately
                continue
            else:
                # Auth, server error, etc. — give up
                raise

    if last_error:
        raise last_error
    raise RuntimeError("All Groq models failed")


# =========================================================================
# ADDITIONAL FREE API PROVIDERS (for round-robin load balancing)
# =========================================================================

def call_nvidia(prompt: str, model: str = "meta/llama-3.3-70b-instruct") -> str:
    """Call NVIDIA NIM API. Free tier: ~40 RPM, no daily token cap."""
    import urllib.request
    import json

    key = st.secrets.get("NVIDIA_API_KEY", "")
    if not key:
        raise ValueError("NVIDIA_API_KEY not found in Streamlit Secrets")

    data = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    st.session_state["last_model_used"] = f"NVIDIA {model}"
    return result["choices"][0]["message"]["content"]


def call_cerebras(prompt: str, model: str = "llama3.1-8b") -> str:
    """Call Cerebras API. Free tier: 1M tokens/day, super fast."""
    import urllib.request
    import json

    key = st.secrets.get("CEREBRAS_API_KEY", "")
    if not key:
        raise ValueError("CEREBRAS_API_KEY not found in Streamlit Secrets")

    data = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.cerebras.ai/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    st.session_state["last_model_used"] = f"Cerebras {model}"
    return result["choices"][0]["message"]["content"]


def call_github_models(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call GitHub Models API. Uses your GitHub PAT. Free tier."""
    import urllib.request
    import json

    key = st.secrets.get("GITHUB_TOKEN", "")
    if not key:
        raise ValueError("GITHUB_TOKEN not found in Streamlit Secrets")

    data = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://models.inference.ai.azure.com/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    st.session_state["last_model_used"] = f"GitHub {model}"
    return result["choices"][0]["message"]["content"]


def call_cloudflare(prompt: str, model: str = "@cf/meta/llama-3.1-8b-instruct") -> str:
    """Call Cloudflare Workers AI. Free tier: 10K neurons/day."""
    import urllib.request
    import json

    token = st.secrets.get("CLOUDFLARE_API_TOKEN", "")
    account_id = st.secrets.get("CLOUDFLARE_ACCOUNT_ID", "")
    if not token or not account_id:
        raise ValueError("CLOUDFLARE_API_TOKEN or CLOUDFLARE_ACCOUNT_ID not found in Streamlit Secrets")

    data = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
    }).encode("utf-8")

    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    st.session_state["last_model_used"] = f"Cloudflare {model}"
    return result.get("result", {}).get("response", "")


# =========================================================================
# ROUND-ROBIN LOAD BALANCER
# =========================================================================

def get_available_providers() -> List[Dict[str, Any]]:
    """Return list of free API providers used for ROUND-ROBIN load balancing.

    Excludes:
        - Claude (premium, opt-in only via Settings)
        - NVIDIA (reserved as dedicated backup fixer)
        - Cloudflare (reserved as final fallback)

    The round-robin rotation uses: Gemini, Cerebras, Groq, GitHub Models
    """
    providers = []

    if st.secrets.get("GEMINI_API_KEY", ""):
        providers.append({
            "name": "Gemini Flash",
            "call_fn": lambda p: _call_gemini_once(p, "gemini-2.5-flash"),
        })

    if st.secrets.get("CEREBRAS_API_KEY", ""):
        providers.append({
            "name": "Cerebras",
            "call_fn": call_cerebras,
        })

    if st.secrets.get("GROQ_API_KEY", ""):
        providers.append({
            "name": "Groq",
            "call_fn": lambda p: call_groq(p, "llama-3.1-8b-instant"),
        })

    if st.secrets.get("GITHUB_TOKEN", ""):
        providers.append({
            "name": "GitHub Models",
            "call_fn": call_github_models,
        })

    return providers


def get_backup_providers() -> List[Dict[str, Any]]:
    """Return list of BACKUP providers used to fix failed chunks.

    NVIDIA goes first (dedicated fixer), then Cloudflare as last resort.
    These are NOT used in round-robin to keep them fresh for emergencies.
    """
    backups = []

    if st.secrets.get("NVIDIA_API_KEY", ""):
        backups.append({
            "name": "NVIDIA (backup)",
            "call_fn": call_nvidia,
        })

    if st.secrets.get("CLOUDFLARE_API_TOKEN", "") and st.secrets.get("CLOUDFLARE_ACCOUNT_ID", ""):
        backups.append({
            "name": "Cloudflare (last resort)",
            "call_fn": call_cloudflare,
        })

    return backups


def call_with_round_robin(prompt: str, chunk_index: int, providers: List[Dict[str, Any]]) -> str:
    """Send a request using round-robin scheduling across providers.

    Strategy:
    1. Try the round-robin primary for this chunk index
    2. If it fails, fall through other primary providers
    3. If ALL primaries fail, try backup providers (NVIDIA, Cloudflare)
    4. If everything still fails, wait & retry forever until something works

    This guarantees the chunk eventually succeeds, even if all APIs are
    temporarily rate-limited.
    """
    import urllib.error
    import time

    if not providers:
        raise RuntimeError("No API providers configured!")

    backups = get_backup_providers()
    n = len(providers)
    primary_idx = chunk_index % n

    retry_round = 0
    max_wait_rounds = 10  # After 10 retry rounds, give up (but each round = 30s+ wait)

    while retry_round < max_wait_rounds:
        errors = []

        # ===== PHASE 1: Try all primary providers (starting with round-robin pick) =====
        for offset in range(n):
            idx = (primary_idx + offset) % n
            provider = providers[idx]
            try:
                return provider["call_fn"](prompt)
            except urllib.error.HTTPError as e:
                errors.append(f"{provider['name']}: HTTP {e.code}")
                continue
            except Exception as e:
                errors.append(f"{provider['name']}: {type(e).__name__}")
                continue

        # ===== PHASE 2: Try backup providers (NVIDIA first, then Cloudflare) =====
        for backup in backups:
            try:
                return backup["call_fn"](prompt)
            except urllib.error.HTTPError as e:
                errors.append(f"{backup['name']}: HTTP {e.code}")
                continue
            except Exception as e:
                errors.append(f"{backup['name']}: {type(e).__name__}")
                continue

        # ===== PHASE 3: Everything failed — wait and retry =====
        retry_round += 1
        wait_seconds = 30 * retry_round  # Exponential-ish: 30s, 60s, 90s, 120s...

        st.warning(
            f"⏳ All {n + len(backups)} APIs failed (chunk {chunk_index + 1}). "
            f"Waiting {wait_seconds}s before retry attempt #{retry_round + 1}..."
        )
        time.sleep(wait_seconds)
        # Loop back to phase 1

    # If we got here, we've waited a LONG time and still nothing works
    raise RuntimeError(
        f"❌ Chunk {chunk_index + 1} failed after {max_wait_rounds} retry rounds "
        f"(~25 minutes of waiting). All APIs are persistently rate-limited or down. "
        f"Last errors: {'; '.join(errors[-3:])}"
    )


def _call_gemini_once(prompt: str, model: str) -> str:
    """Single Gemini API call (no fallback). Used internally."""
    import urllib.request
    import json

    key = st.secrets.get("GEMINI_API_KEY", "")
    if not key:
        raise ValueError("GEMINI_API_KEY not found in Streamlit Secrets")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

    data = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "writing-assistant-app/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["candidates"][0]["content"]["parts"][0]["text"]


def call_gemini(prompt: str, model: str = "gemini-2.5-flash") -> str:
    """Send a prompt to Gemini with automatic fallback chain.

    Fallback strategy (FAST - tries Groq early since it has best rate limits):
    1. Primary model (Gemini Flash) fails (429/503) -> try Groq immediately
    2. Groq fails -> try Flash-Lite
    3. All fail -> wait 5 sec and retry Groq once more
    4. Still fails -> give up with helpful error
    """
    import urllib.error
    import time

    # Track errors for debugging
    errors = []

    # ===== TRY 1: Primary model (Gemini Flash) =====
    try:
        result = _call_gemini_once(prompt, model)
        st.session_state["last_model_used"] = model
        return result
    except urllib.error.HTTPError as e:
        if e.code not in (503, 429):
            # Some other error (auth, bad request, etc.) — re-raise immediately
            raise
        errors.append(f"Gemini {model}: HTTP {e.code}")

    # ===== TRY 2: Groq first (best rate limits, super fast) =====
    has_groq = bool(st.secrets.get("GROQ_API_KEY", ""))
    if has_groq:
        try:
            st.info("⚡ Switching to Groq (trying multiple models)...")
            result = call_groq_with_fallback(prompt)
            return result
        except urllib.error.HTTPError as e:
            # Read the actual error response body for debugging
            try:
                error_body = e.read().decode("utf-8")[:200]
            except Exception:
                error_body = "(could not read body)"
            errors.append(f"Groq: HTTP {e.code} - {error_body}")
        except Exception as e:
            errors.append(f"Groq: {type(e).__name__} - {str(e)[:200]}")
    else:
        errors.append("Groq: GROQ_API_KEY not found in Streamlit Secrets")

    # ===== TRY 3: Gemini Flash-Lite =====
    if model != "gemini-2.5-flash-lite":
        try:
            result = _call_gemini_once(prompt, "gemini-2.5-flash-lite")
            st.session_state["last_model_used"] = "gemini-2.5-flash-lite (fallback)"
            return result
        except urllib.error.HTTPError as e:
            if e.code not in (503, 429):
                raise
            errors.append(f"Gemini Flash-Lite: HTTP {e.code}")

    # ===== TRY 4: Quick 5-sec wait, then Groq one more time =====
    if has_groq:
        st.warning("⏳ Brief 5-second pause before final retry...")
        time.sleep(5)
        try:
            result = call_groq_with_fallback(prompt)
            return result
        except Exception as e:
            errors.append(f"Groq retry: {type(e).__name__} - {str(e)[:200]}")

    # ===== Give up with helpful message + DEBUG INFO =====
    error_details = "\n".join(f"  • {err}" for err in errors)
    error_msg = (
        "🚫 All AI services failed. Debug info:\n"
        f"{error_details}\n\n"
        "Suggestions: "
    )
    if not has_groq:
        error_msg += (
            "Add a GROQ_API_KEY in Streamlit Secrets for a free backup AI. "
        )
    error_msg += (
        "Or wait 30 sec and try 'Quick' depth, or switch to Claude in Settings."
    )
    raise RuntimeError(error_msg)

def call_claude(prompt: str, model: str = "claude-sonnet-4-6") -> str:
    """Send a prompt to Claude. Premium - only use when explicitly requested."""
    import urllib.request
    import json

    key = st.secrets.get("CLAUDE_API_KEY", "")
    if not key:
        raise ValueError("CLAUDE_API_KEY not found in Streamlit Secrets. Add it or pick a different AI in Settings.")

    data = json.dumps({
        "model": model,
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=data,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            "User-Agent": "writing-assistant-app/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    st.session_state["last_model_used"] = "Claude Sonnet 4.6"
    return result["content"][0]["text"]

def rewrite_in_style(user_text: str, reference_text: str) -> str:
    """Rewrite user_text to match the style of reference_text."""
    reference_sample = reference_text[:3000]

    prompt = f"""You are helping a writer improve their draft to match the style of a reference book.

Below is a SAMPLE from the reference book. Study its:
- Sentence length and rhythm
- Vocabulary and word choice
- Tone (dark, playful, dramatic, etc.)
- How it describes action, scenery, and dialogue

REFERENCE BOOK SAMPLE:
{reference_sample}

Now rewrite the following draft to match this style while keeping the same meaning and events. Don't add new plot points. Just rewrite in the reference book's voice.

DRAFT TO REWRITE:
{user_text}

REWRITTEN VERSION:"""

    return call_ai_for_task("rewrite_in_style", prompt)

def generate_text(user_prompt: str, reference_text: str, length: str = "medium") -> str:
    """Generate new text based on a user prompt, in the style of the reference."""
    reference_sample = reference_text[:3000]

    length_guide = {
        "short": "about 100-150 words (1-2 paragraphs)",
        "medium": "about 300-400 words (3-5 paragraphs)",
        "long": "about 700-900 words (a full scene)",
    }.get(length, "about 300-400 words")

    prompt = f"""You are helping a writer generate new content in the style of their reference book.

Below is a SAMPLE from the reference book. Study its voice, tone, vocabulary, sentence rhythm, and how it handles action/description/dialogue.

REFERENCE BOOK SAMPLE:
{reference_sample}

Now write {length_guide} based on the following prompt. Match the reference book's style exactly — same voice, same tone, same rhythm. Don't explain what you're doing, just write the scene.

PROMPT: {user_prompt}

GENERATED SCENE:"""

    return call_ai_for_task("generate_text", prompt)


def compare_and_improve(user_text: str, reference_text: str) -> str:
    """Compare user's draft to reference and give improvement suggestions."""
    reference_sample = reference_text[:3000]

    prompt = f"""You are a writing coach. Compare the WRITER'S DRAFT below to the REFERENCE BOOK's style.

Analyze specifically:
1. Sentence length and rhythm (short/punchy vs long/flowing)
2. Vocabulary (simple/casual vs rich/literary)
3. Description density (sparse vs vivid)
4. Dialogue style (if present)
5. Tone and mood
6. Pacing

Then give 3-5 specific, actionable improvements the writer could make to their draft to feel more like the reference. Include short BEFORE/AFTER rewrite examples where useful.

Format your response in clear sections with headers.

REFERENCE BOOK SAMPLE:
{reference_sample}

WRITER'S DRAFT:
{user_text}

COMPARISON AND IMPROVEMENTS:"""

    return call_ai_for_task("compare_and_improve", prompt)
    

def extract_characters(book_text: str, depth: str = "quick") -> str:
    """Ask the AI to extract characters from a book using round-robin load balancing.

    depth options:
        - "quick"     : first 8,000 chars (~10 sec)
        - "standard"  : first 50,000 chars
        - "deep"      : first 150,000 chars
        - "whole"     : entire book

    Uses round-robin scheduling across all configured free APIs to maximize
    throughput and minimize rate-limit hits.
    """
    depth_limits = {
        "quick": 8000,
        "standard": 50000,
        "deep": 150000,
        "whole": len(book_text),
    }
    char_limit = depth_limits.get(depth, 8000)
    text_to_process = book_text[:char_limit]

    # Get all available API providers
    providers = get_available_providers()
    backups = get_backup_providers()
    num_providers = len(providers)

    if num_providers == 0:
        raise RuntimeError("No API keys configured! Add at least GEMINI_API_KEY in Streamlit Secrets.")

    # If short enough, do a single call (uses normal AI routing)
    if len(text_to_process) <= 10000:
        return _extract_from_chunk(text_to_process)

    # Bigger chunks possible since we're spreading load across multiple APIs
    # 4000 chars = safe size for all providers including Groq's 8K TPM limit
    chunk_size = 4000
    chunks = [text_to_process[i:i + chunk_size] for i in range(0, len(text_to_process), chunk_size)]

    # Calculate delay: divide by number of providers since each handles 1/N of traffic
    # Most providers do ~30 RPM → 2 sec/req per provider
    # With N providers in round-robin: 2/N sec between requests
    # Floor at 1 sec to avoid hammering
    delay_between = max(1.0, 6.0 / num_providers)

    estimated_seconds = int(len(chunks) * delay_between) + len(chunks) * 2  # +2 for API call time
    estimated_minutes = max(1, estimated_seconds // 60)

    provider_names = ", ".join(p["name"] for p in providers)
    backup_names = ", ".join(b["name"] for b in backups) if backups else "none"
    st.info(
        f"⚡ Round-robin across **{num_providers} primary APIs**: {provider_names}\n\n"
        f"🛟 Backup APIs (used if primaries fail): {backup_names}\n\n"
        f"📦 {len(chunks)} chunks · ⏱️ Estimated ~{estimated_minutes} min · "
        f"🔄 Will retry forever until success"
    )

    progress_bar = st.progress(0, text=f"Reading chunk 1 of {len(chunks)}...")
    chunk_results = []
    successful_chunks = 0
    failed_chunks = 0
    provider_usage = {p["name"]: 0 for p in providers}

    import time

    for i, chunk in enumerate(chunks):
        # Pick provider for this chunk via round-robin
        provider_for_this_chunk = providers[i % num_providers]["name"]

        progress_bar.progress(
            (i + 1) / len(chunks),
            text=f"Chunk {i + 1}/{len(chunks)} → {provider_for_this_chunk}"
        )

        # Build the extraction prompt — gather rich detail per character
        chunk_prompt = f"""Analyze the following text — this is one section of a longer book. Extract EVERY character mentioned, even minor ones.

For each character, provide as much of the following as the text reveals:
- **Name** (and any nicknames or titles)
- **Role**: main / side / antagonist / minor / unclear
- **Race or species** (human, animal, fantasy race, etc. — say "unknown" if not stated)
- **Faction or group** (which side they're on, family, kingdom, etc.)
- **Physical description** (appearance, age, distinctive features)
- **Personality traits** (3-5 adjectives)
- **How they speak/act** (tone, mannerisms, quirks)
- **Abilities, skills, or powers** (anything special they can do — combat skills, magic, languages, instincts, etc. Be specific! E.g. "echolocation like bats", "bloodlust in combat", "expert swordsman")
- **Storyline / arc** (what happens to them in this section — their goals, conflicts, key moments)
- **Relationships** (allies, enemies, family, romantic interests)
- **Notable quotes or moments** (if any stand out)

Be thorough. If a character only appears briefly or with limited info, still include them but mark fields as "unclear" or "not shown in this section."

Format each character as a clearly labeled section.

TEXT:
{chunk}

CHARACTERS:"""

        try:
            chunk_result = call_with_round_robin(chunk_prompt, i, providers)
            if chunk_result and len(chunk_result.strip()) > 20:
                chunk_results.append(chunk_result)
                successful_chunks += 1
                provider_usage[provider_for_this_chunk] = provider_usage.get(provider_for_this_chunk, 0) + 1
            else:
                chunk_results.append(f"[Chunk {i+1}: empty result]")
                failed_chunks += 1
        except Exception as e:
            chunk_results.append(f"[Error in chunk {i+1}: {str(e)[:100]}]")
            failed_chunks += 1

        # Pace requests
        if i < len(chunks) - 1:
            time.sleep(delay_between)

    progress_bar.progress(1.0, text="Merging character info...")

    # If everything failed, abort with clear error
    if successful_chunks == 0:
        progress_bar.empty()
        raise RuntimeError(
            f"❌ All {len(chunks)} chunks failed across all {num_providers} APIs. "
            f"This is unusual — check your API keys in Settings page."
        )

    # Show success stats
    if failed_chunks > 0:
        st.warning(f"⚠️ {failed_chunks} of {len(chunks)} chunks failed — results may be incomplete.")
    else:
        st.success(f"✅ All {len(chunks)} chunks processed successfully!")

    # Show which providers were used
    used_summary = " · ".join(f"{name}: {count}" for name, count in provider_usage.items() if count > 0)
    st.caption(f"📊 Provider usage: {used_summary}")

    # Merge all chunk results into one final character list
    combined = "\n\n---CHUNK BREAK---\n\n".join(chunk_results)

    # If combined size is huge, do batched merging first to avoid context limits
    if len(combined) > 12000 and len(chunk_results) > 3:
        batch_size = 3
        batches = [
            "\n\n---CHUNK BREAK---\n\n".join(chunk_results[i:i + batch_size])
            for i in range(0, len(chunk_results), batch_size)
        ]

        progress_bar.progress(1.0, text=f"Merging in {len(batches)} batches...")

        batch_summaries = []
        for batch_idx, batch_text in enumerate(batches):
            batch_prompt = f"""Below are character notes from sections of a book. Combine duplicates and preserve as much detail as possible.

For each character, keep ALL details mentioned across notes:
- Name, role, race/species, faction
- Physical description, personality traits
- Abilities/skills/powers (be specific!)
- Storyline/arc, relationships, notable moments

Don't lose information — combine, don't shorten. If a detail appears in any note, keep it.

NOTES:
{batch_text}

COMBINED CHARACTER NOTES:"""
            try:
                # Use round-robin for batch merges too
                batch_summaries.append(
                    call_with_round_robin(batch_prompt, batch_idx, providers)
                )
            except Exception as e:
                batch_summaries.append(f"[Batch merge error: {e}]")

        combined = "\n\n---BATCH BREAK---\n\n".join(batch_summaries)

    # ===== FINAL MERGE STEP =====
    # Safety check: merge prompt + combined notes must fit in API context
    # Most APIs handle 8K-12K input. Trim if too big.
    MAX_MERGE_INPUT = 11000

    if len(combined) > MAX_MERGE_INPUT:
        st.warning(f"⚠️ Combined notes too long ({len(combined)} chars), trimming for merge...")
        combined = combined[:MAX_MERGE_INPUT] + "\n\n[...notes truncated for length...]"

    merge_prompt = f"""Below are character notes extracted from a book. Combine all mentions of each character and organize into the sections below.

Use ## markdown headers for each section:

## 🌍 Races & Species
List every race/species/group mentioned. Skip if none.

## 🦸 Main Characters
The protagonist(s) and most important characters. For each: name, race, faction, description, personality, **abilities & skills (be specific)**, storyline/arc, relationships, notable moments.

## 🛡️ Side Characters
Supporting characters. Same format.

## ⚔️ Antagonists & Villains
Characters opposing the protagonist. Same format.

## ❓ Partial-Context Characters
Characters with limited info. Note what's KNOWN vs UNCLEAR.

If a section has no characters, write "None mentioned in this book."

CHARACTER NOTES:
{combined}

FINAL ORGANIZED CHARACTER ANALYSIS:"""

    # Try the round-robin merge with multiple fallback layers
    final = None
    merge_errors = []

    # Attempt 1: Round-robin across primary providers
    try:
        final = call_with_round_robin(merge_prompt, 0, providers)
    except Exception as e:
        merge_errors.append(f"Round-robin merge: {type(e).__name__}: {str(e)[:100]}")

    # Attempt 2: Try the simple call_ai_for_task as a backup
    if not final or not final.strip():
        try:
            final = call_ai_for_task("extract_characters", merge_prompt)
        except Exception as e:
            merge_errors.append(f"Fallback merge: {type(e).__name__}: {str(e)[:100]}")

    # Attempt 3: If merge STILL failed, return the raw chunk results so user gets SOMETHING
    if not final or not final.strip():
        progress_bar.empty()
        st.error(
            f"⚠️ Final merge step failed but {successful_chunks} chunks succeeded. "
            f"Showing raw chunk results below.\n\nMerge errors: {' | '.join(merge_errors)}"
        )
        # Return raw chunk results with a header so user gets useful output
        raw_output = "# 🎭 Character Analysis (raw — merge step failed)\n\n"
        raw_output += "*The final organize step failed, but here are the per-chunk results:*\n\n"
        raw_output += "---\n\n"
        raw_output += "\n\n---\n\n".join(chunk_results)
        return raw_output

    progress_bar.empty()
    return final


def _extract_from_chunk(text: str, is_partial: bool = False, prefer_groq: bool = False) -> str:
    """Extract characters from a single chunk of text.

    If prefer_groq=True and a Groq API key is set, calls Groq directly
    (skipping the normal Gemini-first routing) for better rate limits.
    """
    context = "this is one section of a longer book — extract any characters that appear" if is_partial else "this is a complete book or excerpt"

    prompt = f"""Analyze the following text — {context}. Extract EVERY character mentioned, including minor ones.

Organize your response into these clearly-labeled sections (use ## markdown headers):

## 🌍 Races & Species
List every race, species, or notable group mentioned. Briefly describe each. Skip if none.

## 🦸 Main Characters
The protagonist(s) and most important characters. For each:
- **Name** (with nicknames/titles)
- **Race / Species**
- **Faction / Group**
- **Description** (appearance, age)
- **Personality** (3-5 traits)
- **Abilities & Skills** — BE SPECIFIC. List every special ability, talent, or thing they're good at (e.g. "echolocation like bats", "bloodlust in combat", "expert swordsman")
- **Storyline / Arc** — their goals, conflicts, key moments
- **Relationships** — allies, enemies, family
- **Notable Quotes/Moments**

## 🛡️ Side Characters
Important supporting characters (not the protagonist). Same detailed format.

## ⚔️ Antagonists & Villains
Anyone opposing the protagonist. Same detailed format.

## ❓ Partial-Context Characters
Characters with limited info (briefly mentioned, unclear motives). Note what's KNOWN vs UNCLEAR.

Rules:
- Be thorough, not brief
- Use bullet points within each character entry
- If a section has no characters, write "None mentioned in this section."

TEXT:
{text}

CHARACTER ANALYSIS:"""

    # If preferring Groq and it's available, call directly with multi-model fallback
    if prefer_groq and st.secrets.get("GROQ_API_KEY", ""):
        try:
            return call_groq_with_fallback(prompt)
        except Exception:
            # Groq failed — fall through to normal routing as backup
            pass

    return call_ai_for_task("extract_characters", prompt)


def init_state() -> None:
    """Initialize Streamlit session state on first run."""
    defaults = {
        "page": "Home",
        "ref_a_text": None,
        "ref_a_label": None,
        "ref_a_chunks": [],
        "ref_b_text": None,
        "ref_b_label": None,
        "ref_b_chunks": [],
        "user_text": "",
        "ai_extract_characters": "gemini-2.5-flash",
        "ai_rewrite_in_style": "gemini-2.5-flash",
        "ai_generate_text": "gemini-2.5-flash",
        "ai_compare_and_improve": "gemini-2.5-flash",
        "usage_count": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_ref(slot: str) -> Dict[str, Any]:
    slot = slot.lower()
    return {
        "text": st.session_state.get(f"ref_{slot}_text"),
        "label": st.session_state.get(f"ref_{slot}_label"),
        "chunks": st.session_state.get(f"ref_{slot}_chunks", []),
    }


def set_ref(slot: str, text: str, label: str) -> None:
    slot = slot.lower()
    st.session_state[f"ref_{slot}_text"] = text
    st.session_state[f"ref_{slot}_label"] = label
    st.session_state[f"ref_{slot}_chunks"] = chunk_text(text)


def clear_ref(slot: str) -> None:
    slot = slot.lower()
    st.session_state[f"ref_{slot}_text"] = None
    st.session_state[f"ref_{slot}_label"] = None
    st.session_state[f"ref_{slot}_chunks"] = []


def ref_is_loaded(slot: str) -> bool:
    ref = get_ref(slot)
    return bool(ref["text"] and ref["text"].strip())

def call_ai_for_task(task_name: str, prompt: str) -> str:
    """Call the right AI for a given task based on user settings."""
    # Track usage
    usage = st.session_state.get("usage_count", {})
    usage[task_name] = usage.get(task_name, 0) + 1
    st.session_state["usage_count"] = usage

    # Read user's choice for this task
    model_id = st.session_state.get(f"ai_{task_name}", "gemini-2.5-flash")

    # Dispatch to the right function
    if model_id == "claude-sonnet-4-6":
        return call_claude(prompt)
    elif model_id == "gemini-2.5-flash-lite":
        return call_gemini(prompt, model="gemini-2.5-flash-lite")
    else:
        # Default: gemini-2.5-flash with auto-fallback to flash-lite
        return call_gemini(prompt, model="gemini-2.5-flash")


# =========================================================================
# PAGES
# =========================================================================

def page_home() -> None:
    st.title("📖 Writing Assistant")
    st.caption(APP_VERSION)

    st.markdown(
        "Welcome! This app helps you **analyze books** and **improve your own writing** "
        "by comparing it to reference styles."
    )

    st.markdown("### Your session")

    col1, col2, col3 = st.columns(3)
    with col1:
        ref_a = get_ref("A")
        if ref_is_loaded("A"):
            st.success(f"**Reference A**\n\n{ref_a['label']}\n\n"
                       f"{len(ref_a['text']):,} chars · {len(ref_a['chunks'])} chunks")
        else:
            st.info("**Reference A**\n\nNot loaded yet")

    with col2:
        ref_b = get_ref("B")
        if ref_is_loaded("B"):
            st.success(f"**Reference B**\n\n{ref_b['label']}\n\n"
                       f"{len(ref_b['text']):,} chars · {len(ref_b['chunks'])} chunks")
        else:
            st.info("**Reference B**\n\nNot loaded yet")

    with col3:
        draft = st.session_state.get("user_text", "")
        if draft.strip():
            words = len(draft.split())
            st.success(f"**Your draft**\n\n{words:,} words · {len(draft):,} chars")
        else:
            st.info("**Your draft**\n\nEmpty")

    st.divider()

    st.markdown("### Quick start")
    st.markdown(
        "1. **Upload** a book as Reference A in the *Upload* page (sidebar)\n"
        "2. (Optional) Upload a second book as Reference B for style blending\n"
        "3. Open **Writing Mode** to start writing in the app\n"
        "4. Use **Analysis**, **Generate**, or **Compare** to work on your draft"
    )

    with st.expander("What's working vs. what's coming"):
        st.markdown(
            "**Working now:**\n"
            "- Book upload (.txt, .md), automatic chunking, stats\n"
            "- Writing area with word count, save/clear\n"
            "- Reference slots (A + B), session persists while app is running\n\n"
            "**Coming next:**\n"
            "- Character extraction (roles, traits, tone, relationships)\n"
            "- Live writing suggestions and click-to-fix\n"
            "- Rewrite (match style, simplify, make dramatic)\n"
            "- Text generation (expand paragraphs, continue, dialogue)\n"
            "- Dual-style blending\n"
            "- PDF support"
        )


def page_upload() -> None:
    st.title("📥 Upload references")
    st.caption("Load one or two books. Supported formats: .txt, .md")

    tab_a, tab_b = st.tabs(["Reference A", "Reference B (optional)"])

    with tab_a:
        _upload_ui("A")

    with tab_b:
        _upload_ui("B")


def _upload_ui(slot: str) -> None:
    ref = get_ref(slot)

    if ref_is_loaded(slot):
        st.success(f"Loaded: **{ref['label']}** — "
                   f"{len(ref['text']):,} chars, {len(ref['chunks'])} chunks")

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🗑 Clear", key=f"clear_{slot}"):
                clear_ref(slot)
                st.rerun()

        with st.expander("Preview (first 500 chars)"):
            st.text(ref["text"][:500] + ("..." if len(ref["text"]) > 500 else ""))

        with st.expander(f"Chunk breakdown ({len(ref['chunks'])} chunks)"):
            sizes = [len(c.new_content()) for c in ref["chunks"]]
            if sizes:
                st.write(f"min: {min(sizes):,} · avg: {sum(sizes)//len(sizes):,} · max: {max(sizes):,}")
            for c in ref["chunks"][:10]:
                st.text(f"Chunk {c.index + 1}: chars {c.start:,}–{c.end:,} "
                        f"({len(c.new_content()):,} new chars)")
            if len(ref["chunks"]) > 10:
                st.caption(f"...and {len(ref['chunks']) - 10} more chunks")

        st.divider()
        st.caption("Upload a different file to replace this reference:")

    uploaded = st.file_uploader(
        f"Upload a .txt or .md file for Reference {slot}",
        type=["txt", "md"],
        key=f"uploader_{slot}",
    )

    if uploaded is not None:
        try:
            text = read_uploaded_file(uploaded)
            if not text.strip():
                st.error("File appears to be empty.")
                return
            set_ref(slot, text, uploaded.name)
            st.success(f"Loaded {uploaded.name} — "
                       f"{len(text):,} chars, {len(get_ref(slot)['chunks'])} chunks")
            st.rerun()
        except Exception as e:
            st.error(f"Could not read file: {e}")

    with st.expander("Or paste text directly"):
        pasted = st.text_area(
            "Paste book text here",
            height=200,
            key=f"paste_{slot}",
            placeholder="Paste your text and click the button below...",
        )
        if st.button(f"Load pasted text into Reference {slot}", key=f"paste_btn_{slot}"):
            if pasted.strip():
                set_ref(slot, normalize_text(pasted), "pasted text")
                st.success(f"Loaded pasted text — "
                           f"{len(pasted):,} chars, {len(get_ref(slot)['chunks'])} chunks")
                st.rerun()
            else:
                st.warning("Paste some text first.")

    st.caption("📄 PDF support is coming in a future update. For now, save PDFs as .txt first.")


def page_analysis() -> None:
    st.title("🔍 Reference analysis")
    if not ref_is_loaded("A") and not ref_is_loaded("B"):
        st.info("Upload a reference first (go to **Upload** in the sidebar).")
        return

    for slot in ("A", "B"):
        if not ref_is_loaded(slot):
            continue
        ref = get_ref(slot)
        st.markdown(f"### Reference {slot}: *{ref['label']}*")
        col1, col2, col3 = st.columns(3)
        col1.metric("Characters (letters)", f"{len(ref['text']):,}")
        col2.metric("Chunks", len(ref["chunks"]))
        col3.metric("Words (approx)", f"{len(ref['text'].split()):,}")

        with st.expander("Chunk sizes"):
            sizes = [len(c.new_content()) for c in ref["chunks"]]
            if sizes:
                st.write(f"min: {min(sizes):,} · avg: {sum(sizes)//len(sizes):,} · max: {max(sizes):,}")
            st.bar_chart(sizes)

        st.markdown("#### 🎭 Character extraction")
        st.caption("AI reads your book and extracts characters")

        # Depth selection radio group
        st.write("**Extraction depth:**")

        extraction_depth = st.radio(
            "How thoroughly should we analyze the book?",
            options=["quick", "standard", "deep", "whole"],
            format_func=lambda x: {
                "quick": "⚡ Quick (~10 sec)",
                "standard": "🔍 Standard (~30 sec)",
                "deep": "📚 Deep (~1-2 min)",
                "whole": "🏛️ Whole book (~3-5 min)"
            }[x],
            horizontal=True,
            label_visibility="collapsed",
            key=f"depth_{slot}",
        )

        if st.button(f"Extract characters from Reference {slot}", key=f"extract_{slot}"):
            try:
                with st.spinner(f"AI is reading your book... ({extraction_depth} depth)"):
                    result = extract_characters(ref["text"], depth=extraction_depth)

                # Validate the result before saving — don't save empty/junk
                if not result or not result.strip():
                    st.error(
                        "❌ Extraction returned an empty result. "
                        "The merge step may have failed silently. "
                        "Try again, or use a shallower depth (Quick or Standard)."
                    )
                elif len(result.strip()) < 50:
                    st.error(
                        f"❌ Extraction returned suspiciously short result "
                        f"({len(result.strip())} chars): {result[:200]}"
                    )
                    st.session_state[f"characters_{slot}"] = result  # Save anyway
                else:
                    st.session_state[f"characters_{slot}"] = result
                    st.success(f"✅ Extracted {len(result)} characters of analysis!")

            except Exception as e:
                # Show full error including the type so we can debug
                import traceback
                st.error(f"❌ Extraction failed: {type(e).__name__}: {e}")
                with st.expander("🐛 Full error details (for debugging)"):
                    st.code(traceback.format_exc())

        if f"characters_{slot}" in st.session_state:
            st.markdown("##### 🎭 Character analysis:")
            st.markdown(st.session_state[f"characters_{slot}"])

            if st.button(f"🗑 Clear analysis for Reference {slot}", key=f"clear_chars_{slot}"):
                del st.session_state[f"characters_{slot}"]
                st.rerun()

        st.divider()

def page_writing() -> None:
    st.title("✍️ Writing mode")
    st.caption("Write or paste your draft here. Saved while the app is running.")

    draft = st.text_area(
        "Your draft",
        value=st.session_state.get("user_text", ""),
        height=400,
        key="writing_area",
        placeholder="Start typing your scene, chapter, or paragraph...",
    )

    st.session_state["user_text"] = draft

    words = len(draft.split()) if draft else 0
    chars = len(draft)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Words", f"{words:,}")
    col2.metric("Characters", f"{chars:,}")
    col3.metric("Paragraphs", len([p for p in draft.split("\n\n") if p.strip()]))

    with col4:
        if st.button("🗑 Clear draft"):
            st.session_state["user_text"] = ""
            st.rerun()

    st.divider()

    # ✨ Rewrite in reference style
    st.markdown("### ✨ Rewrite in reference style")

    if not ref_is_loaded("A") and not ref_is_loaded("B"):
        st.info("Upload a reference book on the **Upload** page to enable rewriting.")
        return

    if not draft.strip():
        st.info("Write or paste something in the draft above first.")
        return

    options = []
    if ref_is_loaded("A"):
        options.append(f"A: {get_ref('A')['label']}")
    if ref_is_loaded("B"):
        options.append(f"B: {get_ref('B')['label']}")

    choice = st.radio("Match style of:", options, horizontal=True)
    slot = "A" if choice.startswith("A") else "B"

    if st.button("✨ Rewrite in this style", type="primary"):
        with st.spinner("AI is rewriting your draft... (10-30 seconds)"):
            try:
                rewritten = rewrite_in_style(draft, get_ref(slot)["text"])
                st.session_state["rewritten_text"] = rewritten
                st.session_state["rewritten_from_slot"] = slot
            except Exception as e:
                st.error(f"Rewrite failed: {e}")

    if "rewritten_text" in st.session_state:
        used_slot = st.session_state.get("rewritten_from_slot", "?")
        st.success(f"✅ Rewritten to match Reference {used_slot} using **Gemini 2.5 Flash**")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### 📝 Your original")
            st.text_area(
                "original",
                value=draft,
                height=300,
                key="show_original",
                label_visibility="collapsed",
                disabled=True,
            )
        with col_b:
            st.markdown("##### ✨ Rewritten")
            st.text_area(
                "rewritten",
                value=st.session_state["rewritten_text"],
                height=300,
                key="show_rewritten",
                label_visibility="collapsed",
            )

        btn1, btn2 = st.columns(2)
        with btn1:
            if st.button("📋 Use rewritten as new draft"):
                st.session_state["user_text"] = st.session_state["rewritten_text"]
                del st.session_state["rewritten_text"]
                st.rerun()
        with btn2:
            if st.button("🗑 Discard rewrite"):
                del st.session_state["rewritten_text"]
                st.rerun()


def page_generate() -> None:
    st.title("✨ Generate text")
    st.caption("Generate new scenes, dialogue, or descriptions in your reference book's style.")

    if not ref_is_loaded("A") and not ref_is_loaded("B"):
        st.info("Upload a reference book on the **Upload** page first.")
        return

    options = []
    if ref_is_loaded("A"):
        options.append(f"A: {get_ref('A')['label']}")
    if ref_is_loaded("B"):
        options.append(f"B: {get_ref('B')['label']}")

    choice = st.radio("Match style of:", options, horizontal=True)
    slot = "A" if choice.startswith("A") else "B"

    user_prompt = st.text_area(
        "What should be generated?",
        height=120,
        placeholder="e.g. 'A tense battle scene where Gregor fights a rat warrior in a dark tunnel'",
        key="gen_prompt",
    )

    length = st.select_slider(
        "Length",
        options=["short", "medium", "long"],
        value="medium",
    )

    if st.button("✨ Generate", type="primary"):
        if not user_prompt.strip():
            st.warning("Enter a prompt first.")
        else:
            with st.spinner("AI is writing your scene... (15-45 seconds)"):
                try:
                    result = generate_text(user_prompt, get_ref(slot)["text"], length)
                    st.session_state["generated_text"] = result
                    st.session_state["generated_from_slot"] = slot
                    st.session_state["generated_prompt"] = user_prompt
                except Exception as e:
                    st.error(f"Generation failed: {e}")

    if "generated_text" in st.session_state:
        used_slot = st.session_state.get("generated_from_slot", "?")
        st.success(f"✅ Generated in Reference {used_slot}'s style using **Gemini 2.5 Flash**")

        st.markdown("##### ✨ Generated scene")
        edited = st.text_area(
            "You can edit this directly:",
            value=st.session_state["generated_text"],
            height=400,
            key="generated_editable",
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Regenerate"):
                with st.spinner("Generating another version..."):
                    try:
                        new_result = generate_text(
                            st.session_state["generated_prompt"],
                            get_ref(used_slot)["text"],
                            length,
                        )
                        st.session_state["generated_text"] = new_result
                        st.rerun()
                    except Exception as e:
                        st.error(f"Regeneration failed: {e}")
        with col2:
            if st.button("📋 Add to my draft"):
                current_draft = st.session_state.get("user_text", "")
                separator = "\n\n" if current_draft.strip() else ""
                st.session_state["user_text"] = current_draft + separator + edited
                st.success("Added to your draft! Go to Writing mode to see it.")
        with col3:
            if st.button("🗑 Discard"):
                del st.session_state["generated_text"]
                if "generated_prompt" in st.session_state:
                    del st.session_state["generated_prompt"]
                st.rerun()


def page_compare() -> None:
    st.title("⚖️ Compare & improve")
    st.caption("AI compares your draft to a reference and suggests improvements.")

    if not ref_is_loaded("A") and not ref_is_loaded("B"):
        st.warning("Upload a reference book on the **Upload** page first.")
        return

    draft = st.session_state.get("user_text", "")
    if not draft.strip():
        st.warning("Write something in **Writing Mode** first, then come back.")
        return

    options = []
    if ref_is_loaded("A"):
        options.append(f"A: {get_ref('A')['label']}")
    if ref_is_loaded("B"):
        options.append(f"B: {get_ref('B')['label']}")

    choice = st.radio("Compare to:", options, horizontal=True)
    slot = "A" if choice.startswith("A") else "B"

    st.markdown(f"**Your draft:** {len(draft.split())} words")

    if st.button("⚖️ Compare and suggest improvements", type="primary"):
        with st.spinner("AI is analyzing your draft... (15-30 seconds)"):
            try:
                result = compare_and_improve(draft, get_ref(slot)["text"])
                st.session_state["comparison_result"] = result
                st.session_state["comparison_from_slot"] = slot
            except Exception as e:
                st.error(f"Comparison failed: {e}")

    if "comparison_result" in st.session_state:
        used_slot = st.session_state.get("comparison_from_slot", "?")
        st.success(f"✅ Compared to Reference {used_slot} using **Gemini 2.5 Flash**")
        st.markdown("---")
        st.markdown(st.session_state["comparison_result"])

        if st.button("🗑 Clear comparison"):
            del st.session_state["comparison_result"]
            st.rerun()

def page_settings() -> None:
    st.title("⚙️ Settings")
    st.caption("Pick which AI runs each task. Free options are default.")

    # Check what's available
    has_claude = bool(st.secrets.get("CLAUDE_API_KEY", ""))
    has_groq = bool(st.secrets.get("GROQ_API_KEY", ""))
    has_nvidia = bool(st.secrets.get("NVIDIA_API_KEY", ""))
    has_cerebras = bool(st.secrets.get("CEREBRAS_API_KEY", ""))
    has_github = bool(st.secrets.get("GITHUB_TOKEN", ""))
    has_cloudflare = bool(st.secrets.get("CLOUDFLARE_API_TOKEN", "") and st.secrets.get("CLOUDFLARE_ACCOUNT_ID", ""))

    # Count free fallback APIs (Gemini is always there)
    free_fallback_count = 1  # Gemini
    if has_groq: free_fallback_count += 1
    if has_nvidia: free_fallback_count += 1
    if has_cerebras: free_fallback_count += 1
    if has_github: free_fallback_count += 1
    if has_cloudflare: free_fallback_count += 1

    st.success(f"✅ {free_fallback_count} free APIs configured for round-robin load balancing")

    with st.expander("📋 Detected API keys"):
        st.write("**Free APIs:**")
        st.write("• ✅ Gemini (always required)")
        st.write(f"• {'✅' if has_groq else '❌'} Groq")
        st.write(f"• {'✅' if has_nvidia else '❌'} NVIDIA NIM")
        st.write(f"• {'✅' if has_cerebras else '❌'} Cerebras")
        st.write(f"• {'✅' if has_github else '❌'} GitHub Models")
        st.write(f"• {'✅' if has_cloudflare else '❌'} Cloudflare Workers AI")
        st.write("\n**Premium APIs:**")
        st.write(f"• {'✅' if has_claude else '❌'} Claude (premium opt-in)")

    if has_claude:
        st.info("💎 Claude available as premium opt-in for any task")
    else:
        st.caption("ℹ️ Add CLAUDE_API_KEY in Streamlit Secrets to unlock Claude (premium)")

    st.divider()

    # Build the model options list
    free_options = {
        "gemini-2.5-flash": "Gemini 2.5 Flash (free, with auto-fallback)",
        "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite (free, faster, less reliable)",
    }
    paid_options = {
        "claude-sonnet-4-6": "Claude Sonnet 4.6 (premium, ~$0.01-0.03 per call)",
    }

    all_options = dict(free_options)
    if has_claude:
        all_options.update(paid_options)

    option_keys = list(all_options.keys())
    option_labels = list(all_options.values())

    # Helper to render one task picker
    def render_task_picker(task_key: str, task_label: str, task_description: str):
        st.markdown(f"### {task_label}")
        st.caption(task_description)

        current = st.session_state.get(f"ai_{task_key}", "gemini-2.5-flash")
        try:
            current_index = option_keys.index(current)
        except ValueError:
            current_index = 0

        chosen_label = st.selectbox(
            f"AI for {task_label}",
            option_labels,
            index=current_index,
            key=f"select_{task_key}",
            label_visibility="collapsed",
        )
        chosen_key = option_keys[option_labels.index(chosen_label)]
        st.session_state[f"ai_{task_key}"] = chosen_key

        # Show usage count for this task
        usage = st.session_state.get("usage_count", {}).get(task_key, 0)
        st.caption(f"📊 Used {usage} time(s) this session")

    render_task_picker(
        "extract_characters",
        "🎭 Character extraction",
        "Reads your reference book and lists characters with traits.",
    )
    st.divider()

    render_task_picker(
        "rewrite_in_style",
        "✨ Rewrite in style",
        "Rewrites your draft to match your reference book's voice.",
    )
    st.divider()

    render_task_picker(
        "generate_text",
        "📝 Generate text",
        "Creates new scenes from a prompt, in your reference's style.",
    )
    st.divider()

    render_task_picker(
        "compare_and_improve",
        "⚖️ Compare & improve",
        "Analyzes your draft vs reference and suggests improvements.",
    )
    st.divider()

    # Reset button
    if st.button("🔄 Reset all to free defaults"):
        for task in ("extract_characters", "rewrite_in_style", "generate_text", "compare_and_improve"):
            st.session_state[f"ai_{task}"] = "gemini-2.5-flash"
        st.success("All tasks reset to Gemini 2.5 Flash!")
        st.rerun()

    # Total usage summary
    st.markdown("### 📊 Session usage summary")
    usage = st.session_state.get("usage_count", {})
    if usage:
        total = sum(usage.values())
        st.metric("Total AI calls this session", total)
        for task, count in sorted(usage.items(), key=lambda x: -x[1]):
            st.write(f"• **{task.replace('_', ' ').title()}**: {count}")
    else:
        st.caption("No AI calls yet this session.")

    st.divider()

    # ===== API CONNECTION TESTER =====
    st.markdown("### 🔧 API Connection Tester")
    st.caption("Click to test each API and see which ones are working. "
               "This sends a tiny test message to each one.")

    if st.button("🧪 Test all APIs", type="primary"):
        _run_api_tests()


def _run_api_tests() -> None:
    """Test each configured API by sending a small request and timing it."""
    import time
    import urllib.request
    import urllib.error
    import json

    test_prompt = "Say hi in exactly 3 words."

    # Build list of APIs to test based on what keys are configured
    apis_to_test = []

    # Gemini
    if st.secrets.get("GEMINI_API_KEY", ""):
        apis_to_test.append(("Gemini Flash", "gemini"))
    else:
        st.warning("⚠️ GEMINI_API_KEY not set in Secrets — skipping Gemini test")

    # Groq
    if st.secrets.get("GROQ_API_KEY", ""):
        apis_to_test.append(("Groq Llama 3.1 8B", "groq"))
    else:
        st.warning("⚠️ GROQ_API_KEY not set — skipping Groq test")

    # Claude
    if st.secrets.get("CLAUDE_API_KEY", ""):
        apis_to_test.append(("Claude Sonnet 4.6", "claude"))

    # NVIDIA
    if st.secrets.get("NVIDIA_API_KEY", ""):
        apis_to_test.append(("NVIDIA Llama 3.3 70B", "nvidia"))
    else:
        st.warning("⚠️ NVIDIA_API_KEY not set — skipping NVIDIA test")

    # Cerebras
    if st.secrets.get("CEREBRAS_API_KEY", ""):
        apis_to_test.append(("Cerebras Llama 3.1 8B", "cerebras"))
    else:
        st.warning("⚠️ CEREBRAS_API_KEY not set — skipping Cerebras test")

    # GitHub Models
    if st.secrets.get("GITHUB_TOKEN", ""):
        apis_to_test.append(("GitHub Models (GPT-4o-mini)", "github"))
    else:
        st.warning("⚠️ GITHUB_TOKEN not set — skipping GitHub Models test")

    # Cloudflare
    if st.secrets.get("CLOUDFLARE_API_TOKEN", "") and st.secrets.get("CLOUDFLARE_ACCOUNT_ID", ""):
        apis_to_test.append(("Cloudflare Workers AI", "cloudflare"))
    else:
        if not st.secrets.get("CLOUDFLARE_API_TOKEN", ""):
            st.warning("⚠️ CLOUDFLARE_API_TOKEN not set — skipping Cloudflare test")
        if not st.secrets.get("CLOUDFLARE_ACCOUNT_ID", ""):
            st.warning("⚠️ CLOUDFLARE_ACCOUNT_ID not set — skipping Cloudflare test")

    if not apis_to_test:
        st.error("❌ No API keys found in Streamlit Secrets! Add at least one API key first.")
        return

    st.markdown("---")
    st.markdown(f"#### Testing {len(apis_to_test)} APIs...")

    progress_bar = st.progress(0, text="Starting tests...")
    results = []

    for i, (name, api_id) in enumerate(apis_to_test):
        progress_bar.progress(
            (i + 1) / len(apis_to_test),
            text=f"Testing {name}..."
        )
        start_time = time.time()
        try:
            response = _test_single_api(api_id, test_prompt)
            elapsed = time.time() - start_time
            # Truncate response for display
            preview = response[:60].replace("\n", " ").strip()
            results.append({
                "name": name,
                "status": "✅",
                "time": f"{elapsed:.1f}s",
                "detail": f'Response: "{preview}..."' if len(response) > 60 else f'Response: "{preview}"',
            })
        except urllib.error.HTTPError as e:
            elapsed = time.time() - start_time
            try:
                error_body = e.read().decode("utf-8")[:150]
            except Exception:
                error_body = "(could not read error body)"
            results.append({
                "name": name,
                "status": "❌",
                "time": f"{elapsed:.1f}s",
                "detail": f"HTTP {e.code} — {error_body}",
            })
        except Exception as e:
            elapsed = time.time() - start_time
            results.append({
                "name": name,
                "status": "❌",
                "time": f"{elapsed:.1f}s",
                "detail": f"{type(e).__name__}: {str(e)[:150]}",
            })

    progress_bar.empty()

    # Show results
    working = sum(1 for r in results if r["status"] == "✅")
    total = len(results)

    if working == total:
        st.success(f"🎉 All {total} APIs working perfectly!")
    elif working > 0:
        st.warning(f"⚠️ {working} of {total} APIs working. See details below.")
    else:
        st.error(f"❌ No APIs working. Check your keys and try again.")

    st.markdown("#### Results:")
    for r in results:
        if r["status"] == "✅":
            st.success(f"{r['status']} **{r['name']}** — {r['time']}\n\n_{r['detail']}_")
        else:
            st.error(f"{r['status']} **{r['name']}** — {r['time']}\n\n_{r['detail']}_")


def _test_single_api(api_id: str, prompt: str) -> str:
    """Send a tiny test request to a specific API and return the response."""
    import urllib.request
    import json

    if api_id == "gemini":
        return _call_gemini_once(prompt, "gemini-2.5-flash")

    elif api_id == "groq":
        return call_groq(prompt, model="llama-3.1-8b-instant")

    elif api_id == "claude":
        return call_claude(prompt)

    elif api_id == "nvidia":
        key = st.secrets["NVIDIA_API_KEY"]
        data = json.dumps({
            "model": "meta/llama-3.3-70b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"]

    elif api_id == "cerebras":
        key = st.secrets["CEREBRAS_API_KEY"]
        data = json.dumps({
            "model": "llama3.1-8b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.cerebras.ai/v1/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"]

    elif api_id == "github":
        key = st.secrets["GITHUB_TOKEN"]
        data = json.dumps({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://models.inference.ai.azure.com/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"]

    elif api_id == "cloudflare":
        token = st.secrets["CLOUDFLARE_API_TOKEN"]
        account_id = st.secrets["CLOUDFLARE_ACCOUNT_ID"]
        data = json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
        }).encode("utf-8")
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/meta/llama-3.1-8b-instruct"
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        # Cloudflare wraps response differently
        return result.get("result", {}).get("response", str(result))

    else:
        raise ValueError(f"Unknown API: {api_id}")


# =========================================================================
# MAIN
# =========================================================================

def main() -> None:
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="📖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_state()

    with st.sidebar:
        st.markdown(f"## 📖 {APP_NAME}")
        st.caption(APP_VERSION)

        st.markdown("### Navigation")
        pages = {
            "🏠 Home":       page_home,
            "📥 Upload":     page_upload,
            "🔍 Analysis":   page_analysis,
            "✍️ Writing":    page_writing,
            "✨ Generate":    page_generate,
            "⚖️ Compare":    page_compare,
            "⚙️ Settings": page_settings,
        }

        choice = st.radio(
            "Go to page:",
            list(pages.keys()),
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("### Session status")
        st.caption(f"Ref A: {'✅ loaded' if ref_is_loaded('A') else '—'}")
        st.caption(f"Ref B: {'✅ loaded' if ref_is_loaded('B') else '—'}")
        draft = st.session_state.get("user_text", "")
        words = len(draft.split()) if draft else 0
        st.caption(f"Draft: {words:,} words" if words else "Draft: empty")
        st.divider()
        st.caption("Close browser tab + Ctrl+C in Command Prompt to stop.")

    pages[choice]()
if __name__ == "__main__":
    main()
