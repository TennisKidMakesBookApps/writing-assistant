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


def call_gemini(prompt: str, model: str = "gemini-2.5-flash") -> str:
    """Send a prompt to Google Gemini and return the response text."""
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

    return call_gemini(prompt)
    

def extract_characters(book_text: str) -> str:
    """Ask the AI to extract characters from a book."""
    sample = book_text[:8000]

    prompt = f"""Analyze the following text from a book and extract all characters.

For each character, provide:
- Name
- Role (main character / side character / antagonist / minor)
- Personality traits (2-4 adjectives)
- How they speak or act (tone)
- Their relationships with other characters (if any)

Format each character as a clear section with headers.

TEXT:
{sample}

CHARACTERS:"""

    return call_gemini(prompt)


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
        st.caption("AI reads the first ~8,000 characters of your book and extracts characters")

        if st.button(f"Extract characters from Reference {slot}", key=f"extract_{slot}"):
            with st.spinner("AI is reading your book... (10-30 seconds)"):
                try:
                    result = extract_characters(ref["text"])
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

    if not ref_is_loaded("A"):
        st.warning("Upload Reference A first to enable comparison.")
        return

    if not st.session_state.get("user_text", "").strip():
        st.warning("Write something in **Writing Mode** first, then come back.")
        return

    st.markdown(
        "This page will compare your draft to your reference(s) and suggest improvements:"
    )
    st.markdown(
        "- Sentence length and rhythm vs. reference\n"
        "- Vocabulary differences\n"
        "- Description density\n"
        "- Dialogue style match\n"
        "- Rewrite examples: Original / Improved / Author A / Author B / Blended"
    )

    st.divider()
    st.info("🚧 Comparison engine coming next. The data is ready — just needs the analysis logic.")


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
