"""
app.py — HomeFirst Vernacular Loan Counselor
Streamlit UI — Phase 4

Layout:
  Left column  : Audio recorder + conversation transcript
  Right column : Real-time debug panel (LLM internal state)

Features:
  - Push-to-talk audio recorder (streamlit-audiorec)
  - Language detection + lock display
  - Live entity extraction viewer
  - Tool call indicator
  - Lead intent score meter
  - Human handoff alert
  - Full conversation transcript
  - Session reset
"""

import json
import logging
import os
import sys
import time

logger = logging.getLogger(__name__)

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# ── Streamlit page config (must be first Streamlit call) ─────────────────────
st.set_page_config(
    page_title  = "HomeFirst — Ghar Mitra",
    page_icon   = "🏠",
    layout      = "wide",
    initial_sidebar_state = "collapsed",
)

# ── Imports (after path setup) ────────────────────────────────────────────────
from agent.brain import LoanCounselorBrain, TurnResult
from voice.stt   import transcribe
from voice.tts   import speak

# Try importing audio recorder — graceful fallback if not installed
try:
    from st_audiorec import st_audiorec
    AUDIO_RECORDER_AVAILABLE = True
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — clean, professional, HomeFirst brand-inspired
# ─────────────────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    /* ── Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

    /* ── Root variables ── */
    :root {
        --primary:      #1B4F72;
        --primary-light:#2E86C1;
        --accent:       #F39C12;
        --success:      #1E8449;
        --danger:       #C0392B;
        --warning:      #D68910;
        --bg:           #F4F6F8;
        --card:         #FFFFFF;
        --text:         #1C2833;
        --text-muted:   #717D7E;
        --border:       #D5D8DC;
        --handoff:      #6C3483;
    }

    /* ── Global ── */
    html, body, .stApp {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--bg);
        color: var(--text);
    }

    /* ── Header banner ── */
    .hf-header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
        color: white;
        padding: 18px 28px;
        border-radius: 12px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 15px rgba(27,79,114,0.25);
    }
    .hf-header h1 {
        font-family: 'DM Serif Display', serif;
        font-size: 1.7rem;
        margin: 0;
        letter-spacing: -0.3px;
    }
    .hf-header p {
        margin: 2px 0 0 0;
        font-size: 0.85rem;
        opacity: 0.85;
    }
    .hf-logo {
        font-size: 2.4rem;
    }

    /* ── Cards ── */
    .hf-card {
        background: var(--card);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 14px;
        border: 1px solid var(--border);
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .hf-card-title {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: var(--text-muted);
        margin-bottom: 10px;
    }

    /* ── Language badge ── */
    .lang-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: var(--primary);
        color: white;
    }
    .lang-badge.locked {
        background: var(--success);
    }
    .lang-badge.unknown {
        background: var(--text-muted);
    }

    /* ── Step tracker ── */
    .step-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        background: #EBF5FB;
        color: var(--primary-light);
        border: 1px solid #AED6F1;
    }

    /* ── Entity grid ── */
    .entity-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        border-bottom: 1px dashed var(--border);
        font-size: 0.83rem;
    }
    .entity-row:last-child { border-bottom: none; }
    .entity-key {
        color: var(--text-muted);
        font-weight: 500;
    }
    .entity-val {
        font-weight: 600;
        color: var(--primary);
    }
    .entity-val.missing {
        color: #BFC9CA;
        font-weight: 400;
        font-style: italic;
    }

    /* ── Intent score bar ── */
    .score-bar-outer {
        height: 10px;
        background: #EBF5FB;
        border-radius: 10px;
        overflow: hidden;
        margin: 8px 0 4px;
    }
    .score-bar-inner {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }

    /* ── Tool badge ── */
    .tool-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 8px;
        font-size: 0.76rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .tool-badge.called {
        background: #EAFAF1;
        color: var(--success);
        border: 1px solid #A9DFBF;
    }
    .tool-badge.none {
        background: #F2F3F4;
        color: var(--text-muted);
        border: 1px solid var(--border);
    }

    /* ── Handoff alert ── */
    .handoff-alert {
        background: linear-gradient(135deg, #6C3483, #9B59B6);
        color: white;
        padding: 16px 20px;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        font-size: 1rem;
        animation: pulse 1.5s infinite;
        box-shadow: 0 4px 15px rgba(108,52,131,0.4);
        margin-bottom: 14px;
    }
    @keyframes pulse {
        0%   { box-shadow: 0 4px 15px rgba(108,52,131,0.4); }
        50%  { box-shadow: 0 4px 25px rgba(108,52,131,0.7); }
        100% { box-shadow: 0 4px 15px rgba(108,52,131,0.4); }
    }

    /* ── Chat messages ── */
    .chat-msg {
        display: flex;
        gap: 10px;
        margin-bottom: 14px;
        animation: fadeIn 0.3s ease;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .chat-avatar {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        flex-shrink: 0;
        margin-top: 2px;
    }
    .chat-avatar.bot  { background: var(--primary); }
    .chat-avatar.user { background: #F0F3F4; }
    .chat-bubble {
        padding: 10px 14px;
        border-radius: 14px;
        font-size: 0.88rem;
        line-height: 1.55;
        max-width: 88%;
    }
    .chat-bubble.bot {
        background: var(--card);
        border: 1px solid var(--border);
        border-top-left-radius: 4px;
        color: var(--text);
    }
    .chat-bubble.user {
        background: var(--primary);
        color: white;
        border-top-right-radius: 4px;
        margin-left: auto;
    }
    .chat-meta {
        font-size: 0.7rem;
        color: var(--text-muted);
        margin-top: 3px;
    }

    /* ── Recorder area ── */
    .recorder-wrap {
        background: linear-gradient(135deg, #EBF5FB, #FDFEFE);
        border: 2px dashed #AED6F1;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        margin-bottom: 16px;
    }
    .recorder-hint {
        font-size: 0.82rem;
        color: var(--text-muted);
        margin-top: 8px;
    }

    /* ── Status dots ── */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-dot.green  { background: var(--success); }
    .status-dot.orange { background: var(--warning); }
    .status-dot.grey   { background: #BFC9CA; }

    /* ── Latency chip ── */
    .latency-chip {
        display: inline-block;
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 10px;
        background: #F2F3F4;
        color: var(--text-muted);
        font-weight: 500;
    }

    /* ── Eligibility result ── */
    .elig-eligible {
        background: #EAFAF1;
        border: 1px solid #A9DFBF;
        color: var(--success);
        padding: 8px 14px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .elig-not-eligible {
        background: #FDEDEC;
        border: 1px solid #F5B7B1;
        color: var(--danger);
        padding: 8px 14px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .elig-unknown {
        background: #F2F3F4;
        border: 1px solid var(--border);
        color: var(--text-muted);
        padding: 8px 14px;
        border-radius: 8px;
        font-size: 0.85rem;
    }

    /* Hide Streamlit default elements */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }

    /* Streamlit button override */
    .stButton > button {
        background: var(--primary);
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        padding: 8px 20px;
        width: 100%;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background: var(--primary-light);
    }

    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_session():
    if "brain" not in st.session_state:
        # ── Attempt to load RAG retriever (graceful fallback if not installed) ──
        rag_retriever = None
        try:
            from rag import build_retriever
            with st.spinner("Loading knowledge base (first run may take ~30s)..."):
                rag_retriever = build_retriever()
            if rag_retriever:
                logger.info("RAG retriever loaded successfully.")
        except Exception as e:
            logger.warning("RAG unavailable: %s — running without FAQ retrieval.", e)

        st.session_state.brain        = LoanCounselorBrain(rag_retriever=rag_retriever)
        st.session_state.rag_enabled  = rag_retriever is not None

    if "messages" not in st.session_state:
        # messages = list of dicts: {role, text, latency_ms, timestamp}
        st.session_state.messages = []

    if "handoff_triggered" not in st.session_state:
        st.session_state.handoff_triggered = False

    if "last_debug" not in st.session_state:
        st.session_state.last_debug = {}

    if "last_tool_results" not in st.session_state:
        st.session_state.last_tool_results = {}

    if "processing" not in st.session_state:
        st.session_state.processing = False

    if "last_audio_key" not in st.session_state:
        st.session_state.last_audio_key = None


# ─────────────────────────────────────────────────────────────────────────────
# Process one turn (STT → Brain → TTS)
# ─────────────────────────────────────────────────────────────────────────────

def process_turn(user_input: str, source: str = "text"):
    """
    Run one full turn: user_input → GPT-4o brain → TTS → update UI state.

    Args:
        user_input : Transcript text (from STT or typed input)
        source     : "voice" or "text"
    """
    if not user_input.strip():
        return

    brain: LoanCounselorBrain = st.session_state.brain

    # Add user message to transcript
    st.session_state.messages.append({
        "role":       "user",
        "text":       user_input,
        "latency_ms": 0,
        "timestamp":  time.strftime("%H:%M:%S"),
        "source":     source,
    })

    # ── Call the brain ────────────────────────────────────────────────────────
    with st.spinner("Ghar Mitra is thinking..."):
        result: TurnResult = brain.chat(user_input)

    # ── Update session state ──────────────────────────────────────────────────
    st.session_state.last_debug        = brain.state.debug_dict()
    st.session_state.last_tool_results = result.tool_results
    st.session_state.handoff_triggered = brain.state.handoff_triggered

    # Add bot reply to transcript
    st.session_state.messages.append({
        "role":       "bot",
        "text":       result.reply_text,
        "latency_ms": result.latency_ms,
        "timestamp":  time.strftime("%H:%M:%S"),
        "source":     "brain",
    })

    # ── TTS ───────────────────────────────────────────────────────────────────
    if result.reply_text:
        with st.spinner("Generating voice response..."):
            tts_result = speak(result.reply_text, brain.state.locked_language)

        if tts_result.has_audio:
            # Store audio in session state — rendered in the main UI
            st.session_state.latest_audio        = tts_result.audio_bytes
            st.session_state.latest_audio_format = (
                "audio/mp3" if tts_result.provider == "elevenlabs" else "audio/wav"
            )
        else:
            st.session_state.latest_audio = None

    st.session_state.processing = False


# ─────────────────────────────────────────────────────────────────────────────
# Debug panel renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_debug_panel():
    debug = st.session_state.last_debug
    brain: LoanCounselorBrain = st.session_state.brain

    st.markdown("### 🔍 Debug Panel")
    st.caption("Real-time LLM internal state — not visible to the user")

    if not debug:
        st.info("Waiting for first conversation turn...")
        return

    # ── RAG status ────────────────────────────────────────────────────────────
    rag_enabled = st.session_state.get("rag_enabled", False)
    st.markdown('<div class="hf-card">', unsafe_allow_html=True)
    st.markdown('<div class="hf-card-title">Knowledge Base (RAG)</div>', unsafe_allow_html=True)
    if rag_enabled:
        st.markdown(
            '<span class="status-dot green"></span>'
            '<span style="font-size:0.82rem;font-weight:600;color:var(--success);">Active</span>'
            '&nbsp;&nbsp;<span style="font-size:0.75rem;color:var(--text-muted);">ChromaDB · 10 FAQ docs</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="status-dot grey"></span>'
            '<span style="font-size:0.82rem;color:var(--text-muted);">Disabled</span>'
            '&nbsp;&nbsp;<span style="font-size:0.75rem;color:var(--text-muted);">'
            'pip install chromadb sentence-transformers</span>',
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Language lock ─────────────────────────────────────────────────────────
    st.markdown('<div class="hf-card">', unsafe_allow_html=True)
    st.markdown('<div class="hf-card-title">Language</div>', unsafe_allow_html=True)

    lang      = debug.get("locked_language", "unknown")
    locked    = debug.get("language_locked", False)
    lang_label = debug.get("language_label", "Unknown")
    badge_cls = "locked" if locked else ("unknown" if lang == "unknown" else "")

    lock_icon = "🔒" if locked else "🔓"
    st.markdown(
        f'<span class="lang-badge {badge_cls}">{lock_icon} {lang_label}</span>'
        f'&nbsp;&nbsp;<span style="font-size:0.78rem;color:var(--text-muted);">'
        f'{"Language locked" if locked else "Detecting..."}</span>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Conversation step ─────────────────────────────────────────────────────
    st.markdown('<div class="hf-card">', unsafe_allow_html=True)
    st.markdown('<div class="hf-card-title">Current Step</div>', unsafe_allow_html=True)
    step = debug.get("current_step", "greet")
    step_icons = {
        "greet": "👋", "collect": "📋", "confirm": "✅",
        "tool_call": "⚙️", "explain": "💬", "faq": "❓", "handoff": "🤝"
    }
    icon = step_icons.get(step, "•")
    st.markdown(
        f'<span class="step-pill">{icon} {step.upper()}</span>'
        f'&nbsp;&nbsp;<span class="latency-chip">Turn {debug.get("user_turn_count", 0)}</span>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Extracted entities ────────────────────────────────────────────────────
    st.markdown('<div class="hf-card">', unsafe_allow_html=True)
    st.markdown('<div class="hf-card-title">Extracted Entities (JSON)</div>', unsafe_allow_html=True)

    entities = debug.get("entities", {})
    missing  = debug.get("missing_fields", [])

    entity_display = {
        "monthly_income":           ("Monthly Income",        "₹"),
        "property_value":           ("Property Value",        "₹"),
        "loan_amount_requested":    ("Loan Requested",        "₹"),
        "employment_status":        ("Employment",            ""),
        "existing_emi_obligations": ("Existing EMI",          "₹"),
        "tenure_years":             ("Tenure",                "yrs"),
    }

    for key, (label, prefix) in entity_display.items():
        val = entities.get(key)
        if val is not None:
            if prefix == "₹":
                display = f"₹{float(val):,.0f}"
            elif key == "tenure_years":
                display = f"{val} years"
            else:
                display = str(val)
            val_html = f'<span class="entity-val">{display}</span>'
        else:
            val_html = '<span class="entity-val missing">not collected yet</span>'

        req_star = ' <span style="color:var(--danger)">*</span>' if key in missing else ""
        st.markdown(
            f'<div class="entity-row">'
            f'<span class="entity-key">{label}{req_star}</span>'
            f'{val_html}'
            f'</div>',
            unsafe_allow_html=True
        )

    if missing:
        st.markdown(
            f'<div style="margin-top:8px;font-size:0.73rem;color:var(--text-muted);">'
            f'<span style="color:var(--danger)">*</span> Still needed: {", ".join(missing)}'
            f'</div>',
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Tool calls ────────────────────────────────────────────────────────────
    st.markdown('<div class="hf-card">', unsafe_allow_html=True)
    st.markdown('<div class="hf-card-title">Tool Calls</div>', unsafe_allow_html=True)

    tool_called = debug.get("tool_called", "none")
    tools_map = {
        "none":             [],
        "calculate_emi":    ["calculate_emi"],
        "check_eligibility":["check_eligibility"],
        "both":             ["calculate_emi", "check_eligibility"],
    }
    called_tools = tools_map.get(tool_called, [])

    for tool in ["calculate_emi", "check_eligibility"]:
        cls  = "called" if tool in called_tools else "none"
        icon = "✓" if tool in called_tools else "–"
        st.markdown(
            f'<span class="tool-badge {cls}">{icon} {tool}()</span>',
            unsafe_allow_html=True
        )

    # Show tool results if available
    tool_results = st.session_state.last_tool_results
    if tool_results:
        with st.expander("View tool outputs", expanded=False):
            st.json(tool_results)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Eligibility status ────────────────────────────────────────────────────
    st.markdown('<div class="hf-card">', unsafe_allow_html=True)
    st.markdown('<div class="hf-card-title">Eligibility Status</div>', unsafe_allow_html=True)

    elig = debug.get("eligibility_status", "unknown")
    if elig == "eligible":
        st.markdown('<div class="elig-eligible">✅ ELIGIBLE</div>', unsafe_allow_html=True)
    elif elig == "not_eligible":
        st.markdown('<div class="elig-not-eligible">❌ NOT ELIGIBLE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="elig-unknown">⏳ Pending tool call...</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Lead intent score ─────────────────────────────────────────────────────
    st.markdown('<div class="hf-card">', unsafe_allow_html=True)
    st.markdown('<div class="hf-card-title">Lead Intent Score</div>', unsafe_allow_html=True)

    score     = debug.get("lead_intent_score", 0)
    score_pct = min(score * 10, 100)

    if score >= 7:
        bar_color = "#1E8449"     # green
        label     = "🔥 High Intent"
    elif score >= 4:
        bar_color = "#D68910"     # amber
        label     = "📈 Medium Intent"
    else:
        bar_color = "#AED6F1"     # light blue
        label     = "🌱 Early Stage"

    st.markdown(
        f'<div style="display:flex;justify-content:space-between;font-size:0.82rem;">'
        f'<span>{label}</span>'
        f'<strong>{score}/10</strong>'
        f'</div>'
        f'<div class="score-bar-outer">'
        f'<div class="score-bar-inner" style="width:{score_pct}%;background:{bar_color};"></div>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Full state JSON (collapsible) ─────────────────────────────────────────
    with st.expander("Full internal state JSON", expanded=False):
        st.json(debug)


# ─────────────────────────────────────────────────────────────────────────────
# Transcript renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_transcript():
    messages = st.session_state.messages

    if not messages:
        st.markdown(
            '<div style="text-align:center;color:var(--text-muted);padding:40px 0;">'
            '🎤 Press Record and start speaking to begin your loan consultation.'
            '</div>',
            unsafe_allow_html=True
        )
        return

    for msg in messages:
        role      = msg["role"]
        text      = msg["text"]
        timestamp = msg.get("timestamp", "")
        latency   = msg.get("latency_ms", 0)

        if role == "user":
            source_icon = "🎤" if msg.get("source") == "voice" else "⌨️"
            st.markdown(
                f'<div class="chat-msg" style="flex-direction:row-reverse;">'
                f'  <div class="chat-avatar user">{source_icon}</div>'
                f'  <div>'
                f'    <div class="chat-bubble user">{text}</div>'
                f'    <div class="chat-meta" style="text-align:right;">{timestamp}</div>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            latency_str = f" · {latency:.0f}ms" if latency else ""
            st.markdown(
                f'<div class="chat-msg">'
                f'  <div class="chat-avatar bot">🏠</div>'
                f'  <div>'
                f'    <div class="chat-bubble bot">{text}</div>'
                f'    <div class="chat-meta">{timestamp}{latency_str}</div>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

def main():
    inject_css()
    init_session()

    brain: LoanCounselorBrain = st.session_state.brain

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="hf-header">'
        '  <div>'
        '    <h1>🏠 Ghar Mitra</h1>'
        '    <p>HomeFirst Finance · Vernacular Loan Counselor · Powered by GPT-4o</p>'
        '  </div>'
        '  <div class="hf-logo">🏡</div>'
        '</div>',
        unsafe_allow_html=True
    )

    # ── Handoff alert (full width, shown when triggered) ──────────────────────
    if st.session_state.handoff_triggered:
        st.markdown(
            '<div class="handoff-alert">'
            '🤝 [HANDOFF TRIGGERED: Routing to Human Relationship Manager]'
            '<br><small>A HomeFirst representative will contact you shortly.</small>'
            '</div>',
            unsafe_allow_html=True
        )

    # ── Two-column layout ─────────────────────────────────────────────────────
    col_chat, col_debug = st.columns([3, 2], gap="large")

    # ════════════════════════════════
    # LEFT — Chat + Recorder
    # ════════════════════════════════
    with col_chat:

        # ── Audio recorder ────────────────────────────────────────────────────
        st.markdown('<div class="hf-card">', unsafe_allow_html=True)
        st.markdown('<div class="hf-card-title">🎙️ Voice Input (Push-to-Talk)</div>', unsafe_allow_html=True)

        lang = brain.state.locked_language
        locked = brain.state.language_locked
        lang_label = brain.state.debug_dict().get("language_label", "Auto-detect")

        st.markdown(
            f'<div style="margin-bottom:10px;font-size:0.82rem;">'
            f'Current language: '
            f'<span class="lang-badge {"locked" if locked else ""}">{"🔒" if locked else "🔓"} {lang_label}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        if AUDIO_RECORDER_AVAILABLE:
            st.markdown('<div class="recorder-wrap">', unsafe_allow_html=True)
            audio_bytes = st_audiorec()
            st.markdown(
                '<div class="recorder-hint">Speak in Hindi, English, Marathi, or Tamil · '
                'Recording auto-stops after silence</div>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

            # Process audio if new recording received
            if audio_bytes is not None and len(audio_bytes) > 0:
                audio_key = hash(bytes(audio_bytes))
                if audio_key != st.session_state.last_audio_key:
                    st.session_state.last_audio_key = audio_key

                    with st.spinner("Transcribing your speech..."):
                        stt_result = transcribe(
                            bytes(audio_bytes),
                            locked_language = brain.state.locked_language,
                            filename        = "audio.wav",
                        )

                    if stt_result.success:
                        st.success(f"📝 Transcribed: *{stt_result.transcript}*")
                        process_turn(stt_result.transcript, source="voice")
                        st.rerun()
                    else:
                        st.error(f"STT failed: {stt_result.error}")
        else:
            st.warning(
                "Audio recorder not installed. "
                "Run: `pip install streamlit-audiorec` then restart the app."
            )

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Text input fallback ───────────────────────────────────────────────
        st.markdown('<div class="hf-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="hf-card-title">⌨️ Text Input (Fallback)</div>',
            unsafe_allow_html=True
        )

        with st.form("text_input_form", clear_on_submit=True):
            user_text = st.text_input(
                label       = "Type your message",
                placeholder = "e.g. Mujhe 20 lakh ka home loan chahiye...",
                label_visibility = "collapsed",
            )
            submitted = st.form_submit_button("Send →", use_container_width=True)

        if submitted and user_text.strip():
            process_turn(user_text.strip(), source="text")
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Latest audio playback ─────────────────────────────────────────────
        if st.session_state.get("latest_audio"):
            st.markdown('<div class="hf-card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="hf-card-title">🔊 Latest Voice Response</div>',
                unsafe_allow_html=True
            )
            fmt = st.session_state.get("latest_audio_format", "audio/mp3")
            st.audio(st.session_state.latest_audio, format=fmt, autoplay=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Transcript ────────────────────────────────────────────────────────
        st.markdown('<div class="hf-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="hf-card-title">💬 Conversation Transcript</div>',
            unsafe_allow_html=True
        )
        render_transcript()
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Reset button ──────────────────────────────────────────────────────
        if st.button("🔄 Start New Consultation", use_container_width=True):
            brain.reset()
            st.session_state.messages          = []
            st.session_state.handoff_triggered = False
            st.session_state.last_debug        = {}
            st.session_state.last_tool_results = {}
            st.session_state.latest_audio      = None
            st.session_state.last_audio_key    = None
            st.rerun()

    # ════════════════════════════════
    # RIGHT — Debug Panel
    # ════════════════════════════════
    with col_debug:
        render_debug_panel()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
