"""
AI Video Assistant — Streamlit UI
Wraps the existing pipeline (process_input -> transcribe_all -> summarize ->
extract_* -> rag_chain) in a stateful, recruiter-friendly web interface.
"""

import time
import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ----------------------------------------------------------------------------
# PAGE CONFIG + GLOBAL STYLE
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --accent: #6C5CE7;
    --accent-2: #00CEC9;
    --bg-card: rgba(255, 255, 255, 0.04);
    --border-soft: rgba(255, 255, 255, 0.08);
}

/* Page background gradient */
.stApp {
    background: radial-gradient(circle at 10% 0%, #1b1530 0%, #0f0c1d 45%, #0a0814 100%);
}

/* Hide default Streamlit chrome we don't want */
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* Hero header */
.hero {
    padding: 1.6rem 2rem;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(108,92,231,0.25), rgba(0,206,201,0.12));
    border: 1px solid var(--border-soft);
    margin-bottom: 1.5rem;
}
.hero h1 {
    font-size: 2.1rem;
    margin: 0;
    background: linear-gradient(90deg, #a29bfe, #00cec9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}
.hero p {
    margin-top: 0.4rem;
    color: #c9c4e0;
    font-size: 0.95rem;
}

/* Cards */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border-soft);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.glass-card h4 {
    margin-top: 0;
    font-size: 1.0rem;
    color: #a29bfe;
    letter-spacing: 0.02em;
}

/* Pills / badges */
.badge {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    background: rgba(108,92,231,0.18);
    color: #a29bfe;
    border: 1px solid rgba(108,92,231,0.35);
    margin-right: 0.4rem;
}

/* Buttons */
.stButton button {
    border-radius: 10px;
    border: 1px solid var(--border-soft);
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    color: white;
    font-weight: 600;
    transition: transform 0.15s ease;
}
.stButton button:hover {
    transform: translateY(-1px);
    border-color: var(--accent-2);
}

/* Chat bubbles */
[data-testid="stChatMessage"] {
    border-radius: 14px;
    border: 1px solid var(--border-soft);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    background: var(--bg-card);
    padding: 0.5rem 1rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #15102a, #0c0a16);
    border-right: 1px solid var(--border-soft);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# SESSION STATE  (this is the part that actually keeps the app from breaking)
# ----------------------------------------------------------------------------
defaults = {
    "pipeline_ready": False,
    "title": "",
    "transcript": "",
    "summary": "",
    "action_items": "",
    "key_decisions": "",
    "open_questions": "",
    "rag_chain": None,
    "chat_history": [],  # list of (role, text)
    "last_source": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_state():
    for k, v in defaults.items():
        st.session_state[k] = v


# ----------------------------------------------------------------------------
# SIDEBAR — INPUT CONTROLS
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Input")
    source_type = st.radio("Source type", ["YouTube URL", "Local file path"], horizontal=False)

    if source_type == "YouTube URL":
        source = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")
    else:
        uploaded = st.file_uploader("Upload audio/video", type=["mp4", "mp3", "wav", "m4a", "mov"])
        source = uploaded.name if uploaded else ""
        if uploaded is not None:
            # Persist upload to disk so process_input can read a real path
            import os
            os.makedirs("uploads", exist_ok=True)
            save_path = os.path.join("uploads", uploaded.name)
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())
            source = save_path

    language = st.selectbox("Language", ["english", "hinglish"], index=0)

    st.markdown("---")
    run_clicked = st.button("🚀 Run Pipeline", use_container_width=True)
    if st.session_state.pipeline_ready:
        st.button("🔄 Start Over", use_container_width=True, on_click=reset_state)

    st.markdown("---")
    st.caption("Built with Streamlit · Whisper-based transcription · RAG chat over your own meeting/video")

# ----------------------------------------------------------------------------
# HERO HEADER
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🎬 AI Video Assistant</h1>
        <p>Turn any meeting recording or YouTube video into a transcript, summary,
        action items, decisions, and a chatbot you can interrogate — in seconds.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# PIPELINE EXECUTION
# ----------------------------------------------------------------------------
if run_clicked:
    if not source:
        st.error("Give me a YouTube URL or upload a file first.")
    else:
        reset_state()
        st.session_state.last_source = source

        progress_box = st.status("Running pipeline…", expanded=True)
        try:
            with progress_box:
                st.write("🔊 Processing input source…")
                chunks = process_input(source)

                st.write("📝 Transcribing audio…")
                transcript = transcribe_all(chunks, language)
                st.session_state.transcript = transcript

                st.write("🏷️ Generating title…")
                st.session_state.title = generate_title(transcript)

                st.write("📋 Summarizing…")
                st.session_state.summary = summarize(transcript)

                st.write("✅ Extracting action items…")
                st.session_state.action_items = extract_action_items(transcript)

                st.write("🔑 Extracting key decisions…")
                st.session_state.key_decisions = extract_key_decisions(transcript)

                st.write("❓ Extracting open questions…")
                st.session_state.open_questions = extract_questions(transcript)

                st.write("🧠 Building chat engine (RAG)…")
                st.session_state.rag_chain = build_rag_chain(transcript)

                st.session_state.pipeline_ready = True
                progress_box.update(label="Done!", state="complete", expanded=False)
        except Exception as e:
            progress_box.update(label="Pipeline failed", state="error")
            st.exception(e)

# ----------------------------------------------------------------------------
# RESULTS
# ----------------------------------------------------------------------------
if st.session_state.pipeline_ready:
    st.markdown(f"## 📌 {st.session_state.title or 'Untitled'}")
    st.markdown(
        f'<span class="badge">Language: {language}</span>'
        f'<span class="badge">Source: {st.session_state.last_source[:60]}</span>',
        unsafe_allow_html=True,
    )

    tab_summary, tab_actions, tab_transcript, tab_chat = st.tabs(
        ["📋 Summary & Insights", "✅ Action Items / Decisions / Questions", "📄 Full Transcript", "💬 Chat"]
    )

    with tab_summary:
        st.markdown('<div class="glass-card"><h4>Summary</h4></div>', unsafe_allow_html=True)
        st.write(st.session_state.summary)

    with tab_actions:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="glass-card"><h4>✅ Action Items</h4></div>', unsafe_allow_html=True)
            st.write(st.session_state.action_items)
        with col2:
            st.markdown('<div class="glass-card"><h4>🔑 Key Decisions</h4></div>', unsafe_allow_html=True)
            st.write(st.session_state.key_decisions)
        with col3:
            st.markdown('<div class="glass-card"><h4>❓ Open Questions</h4></div>', unsafe_allow_html=True)
            st.write(st.session_state.open_questions)

    with tab_transcript:
        st.text_area("Transcript", st.session_state.transcript, height=400)
        st.download_button(
            "⬇️ Download Transcript (.txt)",
            st.session_state.transcript,
            file_name=f"{(st.session_state.title or 'transcript').replace(' ', '_')}.txt",
        )

    with tab_chat:
        st.markdown("Ask anything about the video — answers are grounded in the transcript via RAG.")

        for role, text in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(text)

        user_q = st.chat_input("Ask a question about this video…")
        if user_q:
            st.session_state.chat_history.append(("user", user_q))
            with st.chat_message("user"):
                st.write(user_q)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        answer = ask_question(st.session_state.rag_chain, user_q)
                    except Exception as e:
                        answer = f"⚠️ Error: {e}"
                    st.write(answer)
            st.session_state.chat_history.append(("assistant", answer))

else:
    st.info("👈 Add a YouTube URL or upload a file in the sidebar, then click **Run Pipeline**.")