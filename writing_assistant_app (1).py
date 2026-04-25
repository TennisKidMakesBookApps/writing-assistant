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

    Tries 8B first (highest daily quota), then 20B, then 70B, then 120B.
    Each model has its own daily budget, so if one is exhausted others may work.
    On 413 (payload too large), tries models with bigger context windows.
    """
    import urllib.error

    models_to_try = [
        "llama-3.1-8b-instant",        # Highest daily quota, 128K context
        "openai/gpt-oss-20b",          # Different model family
        "llama-3.3-70b-versatile",     # 100K/day, very smart
        "openai/gpt-oss-120b",         # Large fallback model
    ]

    last_error = None
    for model in models_to_try:
        try:
            return call_groq(prompt, model=model)
        except urllib.error.HTTPError as e:
            if e.code in (429, 413, 400):
                # 429 = rate limit, 413 = payload too large, 400 = model issue
                # Try next model
                last_error = e
                continue
            else:
                # Some other error (auth, etc) — give up
                raise

    # All Groq models exhausted
    if last_error:
        raise last_error
    raise RuntimeError("All Groq models failed")


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
    """Ask the AI to extract characters from a book.

    depth options:
        - "quick"     : first 8,000 chars (~10 sec)   - uses Gemini
        - "standard"  : first 50,000 chars (~30 sec)  - uses Groq (faster!)
        - "deep"      : first 150,000 chars (~1 min)  - uses Groq (faster!)
        - "whole"     : entire book (~3-5 min)         - uses Groq (faster!)
    """
    depth_limits = {
        "quick": 8000,
        "standard": 50000,
        "deep": 150000,
        "whole": len(book_text),
    }
    char_limit = depth_limits.get(depth, 8000)
    text_to_process = book_text[:char_limit]

    # If short enough, do a single call (uses normal AI routing)
    if len(text_to_process) <= 10000:
        return _extract_from_chunk(text_to_process)

    # For chunked extraction, use Groq directly if available
    # Groq has higher rate limits than Gemini and is much faster
    has_groq = bool(st.secrets.get("GROQ_API_KEY", ""))

    # Chunk size: must stay under Groq's 8K tokens/min per-request limit
    # Each char ≈ 0.25 tokens, plus prompt overhead ~500 tokens
    # 4000 chars ≈ 1000 tokens of text + prompt + output buffer = safe
    chunk_size = 4000 if has_groq else 15000
    chunks = [text_to_process[i:i + chunk_size] for i in range(0, len(text_to_process), chunk_size)]

    if has_groq:
        st.info(f"⚡ Using Groq for fast extraction ({len(chunks)} chunks)")

    progress_bar = st.progress(0, text=f"Reading chunk 1 of {len(chunks)}...")
    chunk_results = []

    import time

    for i, chunk in enumerate(chunks):
        progress_bar.progress(
            (i + 1) / len(chunks),
            text=f"Reading chunk {i + 1} of {len(chunks)}..."
        )
        try:
            chunk_text_result = _extract_from_chunk(chunk, is_partial=True, prefer_groq=has_groq)
            chunk_results.append(chunk_text_result)
        except Exception as e:
            chunk_results.append(f"[Error in chunk {i+1}: {e}]")

        # Smart delay based on which API we're using
        # Groq: 30/min limit -> 2 sec delay
        # Gemini: 15/min limit -> 4 sec delay
        if i < len(chunks) - 1:  # Don't sleep after last chunk
            time.sleep(2 if has_groq else 4)

    progress_bar.progress(1.0, text="Merging character info...")

    # Merge all chunk results into one final character list
    # If combined size is too big for one merge call, do it in batches
    combined = "\n\n---CHUNK BREAK---\n\n".join(chunk_results)

    # Estimate: if combined > 12,000 chars, do batched merging to avoid 413
    if len(combined) > 12000 and len(chunk_results) > 3:
        # Merge in batches of 3 chunks at a time
        batch_size = 3
        batches = [
            "\n\n---CHUNK BREAK---\n\n".join(chunk_results[i:i + batch_size])
            for i in range(0, len(chunk_results), batch_size)
        ]

        progress_bar.progress(1.0, text=f"Merging in {len(batches)} batches...")

        batch_summaries = []
        for batch_text in batches:
            batch_prompt = f"""Below are character notes from one section of a book. Combine duplicates and clean up the format.

For each character: name, role, traits, tone, relationships.

NOTES:
{batch_text}

CLEANED CHARACTER LIST:"""
            try:
                if has_groq:
                    batch_summaries.append(call_groq_with_fallback(batch_prompt))
                else:
                    batch_summaries.append(call_ai_for_task("extract_characters", batch_prompt))
            except Exception as e:
                batch_summaries.append(f"[Batch merge error: {e}]")

        combined = "\n\n---BATCH BREAK---\n\n".join(batch_summaries)

    merge_prompt = f"""Below are character notes extracted from different sections of the same book. The same character may appear multiple times.

Merge them into a single clean character list. For each character:
- Combine traits and details across all mentions
- Remove duplicates
- Keep the role (main/side/antagonist/minor) consistent
- Skip any "Error" notes

Format each character with clear headers: name, role, traits, tone, relationships.

CHARACTER NOTES:
{combined}

FINAL MERGED CHARACTER LIST:"""

    # Use Groq for the merge too if available (faster + bypasses Gemini limits)
    if has_groq:
        try:
            final = call_groq_with_fallback(merge_prompt)
        except Exception:
            # Groq failed for merge — fall back to normal routing
            final = call_ai_for_task("extract_characters", merge_prompt)
    else:
        final = call_ai_for_task("extract_characters", merge_prompt)

    progress_bar.empty()
    return final


def _extract_from_chunk(text: str, is_partial: bool = False, prefer_groq: bool = False) -> str:
    """Extract characters from a single chunk of text.

    If prefer_groq=True and a Groq API key is set, calls Groq directly
    (skipping the normal Gemini-first routing) for better rate limits.
    """
    context = "this is one section of a longer book — extract any characters that appear" if is_partial else "this is a book"

    prompt = f"""Analyze the following text — {context}. Extract all characters mentioned.

For each character, provide:
- Name
- Role (main character / side character / antagonist / minor)
- Personality traits (2-4 adjectives)
- How they speak or act (tone)
- Their relationships with other characters (if any)

Format each character as a clear section with headers.

TEXT:
{text}

CHARACTERS:"""

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
            with st.spinner(f"AI is reading your book... ({extraction_depth} depth)"):
                try:
                    result = extract_characters(ref["text"], depth=extraction_depth)
                    st.session_state[f"characters_{slot}"] = result
                except Exception as e:
                    st.error(f"Extraction failed: {e}")

        if f"characters_{slot}" in st.session_state:
            st.success("✅ Extracted using: **Gemini 2.5 Flash** (free tier via Google)")
            st.markdown("##### Characters found:")
            st.markdown(st.session_state[f"characters_{slot}"])

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

    if has_claude:
        st.success("✅ Claude API key detected — premium options available")
    else:
        st.info("ℹ️ Add a CLAUDE_API_KEY in Streamlit Secrets to unlock Claude options")

    if has_groq:
        st.success("✅ Groq API key detected — used as backup when Gemini hits rate limits")
    else:
        st.info("ℹ️ Add a GROQ_API_KEY in Streamlit Secrets for a free backup AI (recommended!)")

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
