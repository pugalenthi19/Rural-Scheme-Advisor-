# ============================================================
# app.py  —  Rural Scheme Advisor  (All Phases)
# ============================================================
# Features implemented:
#   Phase 1 : Q&A history · Relevance scores · Better sources · Streaming
#   Phase 2 : Dark/Light theme · Copy button · PDF download · Feedback
#   Phase 3 : Tamil support · Scheme comparison · PDF summarize · Suggested Qs
#   Phase 4 : Voice input (st.audio_input) · Voice output (gTTS) · Auto ingest
#   Phase 6 : User login · SQLite storage · Analytics dashboard
# ============================================================

import streamlit as st
import os, re, io, uuid, base64, tempfile
from datetime import datetime
from dotenv import load_dotenv

# LangChain / Vector DB
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# Local DB module
from database import (
    init_db, verify_login, register_user,
    save_conversation, log_question, save_feedback, get_analytics,
)

load_dotenv()

# ──────────────────────────────────────────────
# Optional-package flags
# ──────────────────────────────────────────────
HAS_LANGDETECT = False
HAS_GTTS       = False
HAS_FPDF       = False
HAS_SR         = False

try:
    from langdetect import detect as _ld_detect; HAS_LANGDETECT = True
except ImportError: pass

try:
    from gtts import gTTS; HAS_GTTS = True
except ImportError: pass

try:
    from fpdf import FPDF; HAS_FPDF = True
except ImportError: pass

try:
    import speech_recognition as _sr_lib; HAS_SR = True
except ImportError: pass


# ──────────────────────────────────────────────
# Page config (MUST be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Rural Scheme Advisor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

# ──────────────────────────────────────────────
# Session-state defaults
# ──────────────────────────────────────────────
_DEFAULTS = dict(
    logged_in=False, user_id=None, username=None, role=None,
    session_id=str(uuid.uuid4()), messages=[], questions=[],
    dark_mode=False, feedback_given={}, suggested_questions=[],
    show_register=False,
)
for k, v in _DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ──────────────────────────────────────────────
# Theme  (inject once per render)
# ──────────────────────────────────────────────
def _inject_theme():
    if st.session_state.dark_mode:
        bg, fg          = "#141E14", "#E8F5E9"
        sidebar_bg      = "#0F170F"
        card            = "#1E2E1E"
        accent          = "#66BB6A"
        border          = "#2E3B2E"
        btn_bg, btn_fg  = "#388E3C", "#FFFFFF"
        input_bg        = "#1E2E1E"
        tag_bg          = "#2E7D32"
    else:
        bg, fg          = "#F1F8E9", "#1B4B1E"
        sidebar_bg      = "#E8F5E9"
        card            = "#FFFFFF"
        accent          = "#2E7D32"
        border          = "#C8E6C9"
        btn_bg, btn_fg  = "#2E7D32", "#FFFFFF"
        input_bg        = "#FFFFFF"
        tag_bg          = "#A5D6A7"

    st.markdown(f"""
    <style>
    /* ── Global ── */
    .stApp {{ background:{bg}; color:{fg}; font-family:'Segoe UI',sans-serif; }}
    section[data-testid="stSidebar"] {{ background:{sidebar_bg}; border-right:1px solid {border}; }}

    /* ── Chat bubbles ── */
    [data-testid="stChatMessage"] {{
        background:{card}; border:1px solid {border};
        border-radius:14px; padding:14px 18px; margin:6px 0;
    }}

    /* ── Source cards ── */
    .src-card {{
        background:{card}; border-left:4px solid {accent};
        border-radius:8px; padding:10px 14px; margin:6px 0;
        font-size:.88rem;
    }}

    /* ── Metric cards ── */
    .metric-box {{
        background:{card}; border:1px solid {border}; border-radius:12px;
        padding:18px; text-align:center;
    }}
    .metric-box .num  {{ font-size:2rem; font-weight:700; color:{accent}; }}
    .metric-box .lbl  {{ font-size:.8rem; color:{fg}; opacity:.7; margin-top:2px; }}

    /* ── Scheme tags ── */
    .scheme-tag {{
        display:inline-block; background:{tag_bg}; color:{btn_fg};
        padding:2px 10px; border-radius:20px; font-size:.75rem;
        margin:2px; font-weight:600;
    }}

    /* ── Headings ── */
    h1,h2,h3 {{ color:{accent}; }}

    /* ── Buttons ── */
    .stButton>button {{
        background:{btn_bg}; color:{btn_fg};
        border:none; border-radius:8px;
        padding:6px 16px; font-weight:600;
        transition:opacity .2s;
    }}
    .stButton>button:hover {{ opacity:.85; }}

    /* ── Inputs ── */
    .stTextInput>div>div>input,
    .stTextArea textarea {{
        background:{input_bg}; color:{fg};
        border:1px solid {border}; border-radius:8px;
    }}

    /* ── Select boxes ── */
    .stSelectbox>div>div {{
        background:{input_bg}; color:{fg};
        border:1px solid {border};
    }}

    /* ── Divider ── */
    hr {{ border-color:{border}; }}

    /* ── Chat input ── */
    [data-testid="stChatInputContainer"] {{
        background:{card}; border-top:1px solid {border};
    }}

    /* ── Relevance bar ── */
    .rel-high {{ color:#2E7D32; font-weight:700; }}
    .rel-mid  {{ color:#F57F17; font-weight:700; }}
    .rel-low  {{ color:#C62828; font-weight:700; }}
    </style>
    """, unsafe_allow_html=True)

_inject_theme()


# ──────────────────────────────────────────────
# Cached resources
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embeddings…")
def _load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Connecting to vector store…")
def _load_vectordb():
    os.makedirs("vectorstore", exist_ok=True)
    return Chroma(persist_directory="vectorstore", embedding_function=_load_embeddings())

@st.cache_resource(show_spinner="Loading LLM…")
def _load_llm():
    return ChatGroq(model_name="llama-3.3-70b-versatile")


# ──────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────
def detect_language(text: str) -> str:
    """Return 'ta' for Tamil, 'en' otherwise."""
    if re.search(r"[\u0B80-\u0BFF]", text):
        return "ta"
    if HAS_LANGDETECT:
        try:
            l = _ld_detect(text)
            return l if l in ("ta", "en") else "en"
        except Exception:
            pass
    return "en"


def relevance_pct(score: float) -> int:
    """Convert ChromaDB cosine distance → 0-100 %."""
    return max(0, min(100, round((1 - min(score, 1.0)) * 100)))


def rel_class(pct: int) -> str:
    if pct >= 70: return "rel-high"
    if pct >= 40: return "rel-mid"
    return "rel-low"


def build_prompt(question: str, context: str, history: str, lang: str) -> str:
    if lang == "ta":
        system = (
            "நீங்கள் ஒரு நிபுணர் கிராமப்புற திட்ட ஆலோசகர். "
            "கீழே உள்ள ஆவணங்களை மட்டுமே பயன்படுத்தி தமிழிலும் "
            "ஆங்கிலத்திலும் (தொழில்நுட்பச் சொற்களுக்கு) பதில் அளிக்கவும். "
            "தகுதி, நோக்கம், நன்மைகள் மற்றும் விண்ணப்ப நடைமுறை ஆகியவற்றை விளக்கவும்."
        )
    else:
        system = (
            "You are an Expert Rural Scheme Advisor for Indian farmers.\n"
            "Use ONLY the provided context. Structure your answer with clear headings,\n"
            "bullet points for eligibility / benefits / procedure, and plain language.\n"
            "If the answer is not in the context, say: "
            '"I could not find this information in the documents."'
        )
    return f"""{system}

Conversation History:
{history}

Context from Documents:
{context}

Question: {question}

Answer:"""


def text_to_audio(text: str, lang: str = "en") -> bytes | None:
    if not HAS_GTTS:
        return None
    try:
        tts = gTTS(text=text[:1200], lang=lang, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        return buf.getvalue()
    except Exception as e:
        st.warning(f"TTS error: {e}")
        return None


def generate_pdf(question: str, answer: str, sources: list, username: str) -> bytes | None:
    if not HAS_FPDF:
        return None
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_margins(15, 15, 15)

        # Header
        pdf.set_fill_color(46, 125, 50)
        pdf.rect(0, 0, 210, 28, style="F")
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 14, "Rural Scheme Advisor", ln=True, align="C")
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 8,
            f"Generated: {datetime.now():%Y-%m-%d %H:%M}   |   User: {username}",
            ln=True, align="C")
        pdf.ln(8)

        def sec(title: str):
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(46, 125, 50)
            pdf.cell(0, 8, title, ln=True)
            pdf.set_text_color(30, 30, 30)
            pdf.set_font("Helvetica", "", 10)

        sec("Question")
        pdf.multi_cell(0, 6, question.encode("latin-1", "replace").decode("latin-1"))
        pdf.ln(4)

        sec("Answer")
        clean = re.sub(r"[#*`]+", "", answer)
        clean = re.sub(r"\n{3,}", "\n\n", clean)
        pdf.multi_cell(0, 5, clean.encode("latin-1", "replace").decode("latin-1"))
        pdf.ln(4)

        if sources:
            sec("References")
            for s in sources:
                pdf.cell(0, 5,
                    f"- {s}".encode("latin-1", "replace").decode("latin-1"),
                    ln=True)

        return bytes(pdf.output())
    except Exception as e:
        st.error(f"PDF error: {e}")
        return None


def auto_ingest(path: str, vectordb) -> int:
    loader = PyPDFLoader(path)
    docs   = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks   = splitter.split_documents(docs)
    vectordb.add_documents(chunks)
    return len(chunks)


def suggested_questions(llm, question: str, answer: str) -> list[str]:
    try:
        r = llm.invoke(
            f"Based on this Q&A about Indian rural schemes, "
            f"suggest exactly 3 natural follow-up questions (no numbering, one per line).\n"
            f"Q: {question}\nA: {answer[:400]}"
        )
        return [q.strip() for q in r.content.splitlines() if "?" in q][:3]
    except Exception:
        return []


def speech_to_text(audio_bytes: bytes, lang: str = "en-IN") -> str | None:
    if not HAS_SR:
        return None
    try:
        recognizer = _sr_lib.Recognizer()
        buf = io.BytesIO(audio_bytes)
        with _sr_lib.AudioFile(buf) as src:
            audio = recognizer.record(src)
        return recognizer.recognize_google(audio, language=lang)
    except Exception:
        return None


def copy_button_html(text: str, key: str) -> str:
    """Copy button with HTTPS clipboard API + HTTP execCommand fallback."""
    b64 = base64.b64encode(text.encode()).decode()
    return f"""
    <script>
    function doCopy_{key}(){{
        const t = atob('{b64}');
        const btn = document.getElementById('cb_{key}');
        function flash(){{
            if(btn){{
                btn.innerText = '✅ Copied!';
                setTimeout(()=>{{ btn.innerText = '📋 Copy Answer'; }}, 2000);
            }}
        }}
        if(navigator.clipboard && window.isSecureContext){{
            navigator.clipboard.writeText(t).then(flash).catch(()=> fallback(t));
        }} else {{
            fallback(t);
        }}
    }}
    function fallback(t){{
        const ta = document.createElement('textarea');
        ta.value = t;
        ta.style.position = 'fixed';
        ta.style.opacity  = '0';
        document.body.appendChild(ta);
        ta.focus(); ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        const btn = document.getElementById('cb_{key}');
        if(btn){{
            btn.innerText = '✅ Copied!';
            setTimeout(()=>{{ btn.innerText = '📋 Copy Answer'; }}, 2000);
        }}
    }}
    </script>
    <button id="cb_{key}" onclick="doCopy_{key}()"
        style="background:#2E7D32;color:#fff;border:none;padding:7px 18px;
               border-radius:8px;cursor:pointer;font-weight:600;width:100%;
               font-family:sans-serif;font-size:14px;">
        📋 Copy Answer
    </button>
    """


# ──────────────────────────────────────────────
# LOGIN PAGE
# ──────────────────────────────────────────────
def _login_page():
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("""
        <div style="text-align:center;padding:32px 0 16px">
            <div style="font-size:3rem">🌾</div>
            <h1 style="margin:0">Rural Scheme Advisor</h1>
            <p style="opacity:.7">AI-Powered Assistant for Government Rural &amp; Agricultural Schemes</p>
        </div>""", unsafe_allow_html=True)

        if not st.session_state.show_register:
            st.subheader("🔐 Login")
            u = st.text_input("Username", key="li_u")
            p = st.text_input("Password", type="password", key="li_p")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("🚀 Login", use_container_width=True):
                    row = verify_login(u, p)
                    if row:
                        st.session_state.update(
                            logged_in=True, user_id=row[0],
                            username=row[1], role=row[2],
                            session_id=str(uuid.uuid4()),
                            messages=[], questions=[],
                        )
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials")
            with c2:
                if st.button("📝 Register", use_container_width=True):
                    st.session_state.show_register = True
                    st.rerun()

            st.divider()
            st.markdown("""
**Demo accounts:**

| Username | Password | Role |
|----------|----------|------|
| farmer1  | farmer123 | Farmer |
| admin    | admin123  | Officer |
""")
        else:
            st.subheader("📝 Register")
            nu = st.text_input("Username", key="re_u")
            np = st.text_input("Password", type="password", key="re_p")
            cp = st.text_input("Confirm Password", type="password", key="re_cp")
            role = st.selectbox("Role", ["farmer", "officer"])

            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Register", use_container_width=True):
                    if np != cp:
                        st.error("Passwords don't match")
                    elif len(nu) < 3:
                        st.error("Username too short (min 3)")
                    elif len(np) < 6:
                        st.error("Password too short (min 6)")
                    elif register_user(nu, np, role):
                        st.success("✅ Registered! Please log in.")
                        st.session_state.show_register = False
                        st.rerun()
                    else:
                        st.error("Username already taken")
            with c2:
                if st.button("← Back", use_container_width=True):
                    st.session_state.show_register = False
                    st.rerun()


# ──────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────
def _main_app():
    vectordb = _load_vectordb()
    llm      = _load_llm()

    # ── Sidebar ───────────────────────────────
    with st.sidebar:
        st.markdown(f"## 🌾 Rural Advisor")
        role_icon = "👮" if st.session_state.role == "officer" else "🧑‍🌾"
        st.markdown(
            f"{role_icon} **{st.session_state.username}**  "
            f"<span class='scheme-tag'>{st.session_state.role}</span>",
            unsafe_allow_html=True,
        )

        # Theme toggle
        lbl = "☀️ Light Mode" if st.session_state.dark_mode else "🌙 Dark Mode"
        if st.button(lbl, use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

        st.divider()

        # ── PDF Upload with Auto Ingest ──
        st.subheader("📤 Upload PDF")
        upfile = st.file_uploader("Choose a PDF", type=["pdf"], key="sb_pdf")

        if upfile:
            do_summarize = st.checkbox("Auto-Summarize after ingestion", value=True)
            if st.button("🔄 Ingest into Knowledge Base", use_container_width=True):
                os.makedirs("data", exist_ok=True)
                path = os.path.join("data", upfile.name)
                with open(path, "wb") as f:
                    f.write(upfile.getbuffer())
                with st.spinner("📚 Chunking & embedding…"):
                    try:
                        n = auto_ingest(path, vectordb)
                        st.success(f"✅ {n} chunks added from **{upfile.name}**")
                        if do_summarize:
                            with st.spinner("Summarising document…"):
                                loader = PyPDFLoader(path)
                                raw = "\n".join(
                                    [d.page_content for d in loader.load()[:4]]
                                )
                                s = llm.invoke(
                                    "Summarise this government scheme document.\n"
                                    "Give: 1) Objectives 2) Target Beneficiaries "
                                    "3) Key Benefits 4) Application Process\n\n"
                                    f"Document:\n{raw[:3000]}"
                                )
                                st.info(s.content)
                    except Exception as e:
                        st.error(f"Ingest error: {e}")

        st.divider()

        # ── Conversation History ──
        st.subheader("💬 Conversation History")
        msgs = st.session_state.messages
        if msgs:
            for m in reversed(msgs[-12:]):
                icon = "🙋" if m["role"] == "user" else "🤖"
                preview = m["content"][:55] + ("…" if len(m["content"]) > 55 else "")
                st.markdown(f"{icon} _{preview}_")
        else:
            st.caption("No conversation yet.")

        st.divider()

        # ── Session stats ──
        st.subheader("📊 Session Stats")
        st.metric("Questions asked", len(st.session_state.questions))

        st.divider()
        st.subheader("📌 Topics")
        st.markdown("""
<span class="scheme-tag">PM-KISAN</span>
<span class="scheme-tag">KCC</span>
<span class="scheme-tag">eNAM</span>
<span class="scheme-tag">PMFBY</span>
<span class="scheme-tag">MNREGS</span>
<span class="scheme-tag">PMGSY</span>
<span class="scheme-tag">Crop Insurance</span>
""", unsafe_allow_html=True)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑 Clear Chat", use_container_width=True):
                st.session_state.messages          = []
                st.session_state.questions         = []
                st.session_state.suggested_questions = []
                st.rerun()
        with c2:
            if st.button("🚪 Logout", use_container_width=True):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()

    # ── Main header ───────────────────────────
    st.title("🌾 Rural Scheme Advisor")
    st.caption("AI-Powered Assistant for Government Rural & Agricultural Schemes")

    # ── Tabs ─────────────────────────────────
    tab_labels = ["💬 Chat", "📊 Compare Schemes", "📄 Summarise PDF", "📈 Analytics"]
    tabs = st.tabs(tab_labels)
    tab_chat, tab_compare, tab_summarise, tab_analytics = tabs

    # ═══════════════════════════════════════════
    # TAB 1 — CHAT
    # ═══════════════════════════════════════════
    with tab_chat:
        # Display previous messages
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                # Sources expander
                if msg["role"] == "assistant" and msg.get("sources"):
                    with st.expander("📚 Source Documents"):
                        for src in msg["sources"]:
                            st.markdown(
                                f'<div class="src-card">{src}</div>',
                                unsafe_allow_html=True,
                            )

                # Feedback buttons
                if msg["role"] == "assistant":
                    fb_key = f"fb_{i}"
                    if fb_key not in st.session_state.feedback_given:
                        fc1, fc2, _ = st.columns([1, 1, 8])
                        q_idx = i // 2 - 1
                        prev_q = (
                            st.session_state.questions[q_idx]
                            if 0 <= q_idx < len(st.session_state.questions)
                            else ""
                        )
                        with fc1:
                            if st.button("👍", key=f"up_{i}"):
                                save_feedback(st.session_state.user_id, prev_q, 1)
                                st.session_state.feedback_given[fb_key] = "helpful"
                                st.rerun()
                        with fc2:
                            if st.button("👎", key=f"dn_{i}"):
                                save_feedback(st.session_state.user_id, prev_q, 0)
                                st.session_state.feedback_given[fb_key] = "not helpful"
                                st.rerun()
                    else:
                        st.caption(f"Feedback: {st.session_state.feedback_given[fb_key]}")

        # ── Suggested questions ──────────────
        if st.session_state.suggested_questions:
            st.markdown("**💡 Related Questions — click to ask:**")
            sq_cols = st.columns(len(st.session_state.suggested_questions))
            for idx, (col, sq) in enumerate(
                zip(sq_cols, st.session_state.suggested_questions)
            ):
                with col:
                    label = (sq[:52] + "…") if len(sq) > 52 else sq
                    if st.button(label, key=f"sq_{idx}", use_container_width=True):
                        st.session_state["_pending_q"] = sq
                        st.session_state.suggested_questions = []
                        st.rerun()

        # ── Voice input toggle ───────────────
        st.markdown("---")
        voice_on = st.checkbox("🎤 Use Voice Input", key="voice_toggle")
        voice_q  = None

        if voice_on:
            v_lang = st.radio(
                "Voice language",
                ["English (India)", "Tamil"],
                horizontal=True,
                key="v_lang",
            )
            v_code = "ta-IN" if "Tamil" in v_lang else "en-IN"

            try:
                audio_val = st.audio_input("🎙️ Click mic to record your question")
                if audio_val is not None:
                    if HAS_SR:
                        with st.spinner("Converting speech to text…"):
                            voice_q = speech_to_text(audio_val.read(), lang=v_code)
                        if voice_q:
                            st.info(f"📝 Recognised: _{voice_q}_")
                        else:
                            st.warning(
                                "Could not understand audio. "
                                "Please speak clearly or type your question."
                            )
                    else:
                        st.warning(
                            "SpeechRecognition not installed. "
                            "Run: `pip install SpeechRecognition`"
                        )
            except AttributeError:
                st.info("Voice input requires Streamlit ≥ 1.35. Please type your question.")

        # ── Collect question from any source ─
        pending_q = st.session_state.pop("_pending_q", None)
        typed_q   = st.chat_input("Ask about PM-KISAN, eNAM, KCC, PMFBY…")
        question  = pending_q or voice_q or typed_q

        # ── Process question ─────────────────
        if question:
            lang = detect_language(question)
            st.session_state.questions.append(question)
            log_question(st.session_state.user_id, question, lang)
            save_conversation(
                st.session_state.user_id, st.session_state.session_id,
                "user", question,
            )

            # Display user bubble
            with st.chat_message("user"):
                st.markdown(question)

            # ── AI response ──────────────────
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching knowledge base…"):
                    try:
                        results = vectordb.similarity_search_with_score(question, k=5)
                    except Exception:
                        results = []

                    if not results:
                        no_doc = (
                            "⚠️ No documents found in the knowledge base. "
                            "Please upload and ingest PDF files first using the sidebar."
                        )
                        st.warning(no_doc)
                        st.session_state.messages.append(
                            {"role": "user",      "content": question}
                        )
                        st.session_state.messages.append(
                            {"role": "assistant", "content": no_doc}
                        )
                        st.stop()

                    docs   = [r[0] for r in results]
                    scores = [r[1] for r in results]
                    ctx    = "\n\n".join(d.page_content for d in docs)

                    history = "\n".join(
                        f"{m['role'].upper()}: {m['content'][:200]}"
                        for m in st.session_state.messages[-8:]
                    )
                    prompt = build_prompt(question, ctx, history, lang)

                # Stream response
                placeholder  = st.empty()
                full_response = ""
                try:
                    for chunk in llm.stream(prompt):
                        full_response += chunk.content
                        placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)
                except Exception:
                    resp = llm.invoke(prompt)
                    full_response = resp.content
                    placeholder.markdown(full_response)

                # ── Source cards with relevance ──
                source_list = []
                for doc, score in zip(docs, scores):
                    fname = os.path.basename(doc.metadata.get("source", "Unknown"))
                    page  = doc.metadata.get("page", "?")
                    pct   = relevance_pct(score)
                    cls   = rel_class(pct)
                    source_list.append(
                        (fname, page, pct, cls, doc.page_content)
                    )

                # De-dup by file+page
                seen, unique = set(), []
                for item in source_list:
                    k = f"{item[0]}|{item[1]}"
                    if k not in seen:
                        seen.add(k)
                        unique.append(item)

                with st.expander("📚 Source Documents with Relevance Scores"):
                    for idx, (fname, page, pct, cls, content) in enumerate(unique, 1):
                        c_left, c_right = st.columns([4, 1])
                        with c_left:
                            st.markdown(
                                f'<div class="src-card">📄 <b>{fname}</b> &nbsp;·&nbsp; Page {page}</div>',
                                unsafe_allow_html=True,
                            )
                        with c_right:
                            st.markdown(
                                f'<span class="{cls}">{pct}% match</span>',
                                unsafe_allow_html=True,
                            )
                        with st.expander(f"View content — Source {idx}"):
                            st.text_area(
                                "",
                                value=content[:1500],
                                height=140,
                                disabled=True,
                                key=f"sc_{idx}_{len(st.session_state.messages)}",
                            )

                src_labels = [
                    f"📄 {f} | Page {p} | {pct}% match"
                    for f, p, pct, *_ in unique
                ]

                # ── Action row ───────────────
                ba, bb, bc = st.columns(3)

                with ba:
                    uid = str(len(st.session_state.messages))
                    st.components.v1.html(
                        copy_button_html(full_response, uid), height=42
                    )

                with bb:
                    if HAS_FPDF:
                        pdf_bytes = generate_pdf(
                            question, full_response,
                            src_labels, st.session_state.username,
                        )
                        if pdf_bytes:
                            fname_pdf = (
                                f"scheme_report_{datetime.now():%Y%m%d_%H%M}.pdf"
                            )
                            st.download_button(
                                "⬇️ Download PDF",
                                data=pdf_bytes,
                                file_name=fname_pdf,
                                mime="application/pdf",
                                use_container_width=True,
                            )
                    else:
                        st.caption("Install fpdf2 for PDF download")

                with bc:
                    if HAS_GTTS:
                        tts_key = f"tts_{len(st.session_state.messages)}"
                        if st.button("🔊 Listen", key=tts_key, use_container_width=True):
                            audio_lang = "ta" if lang == "ta" else "en"
                            ab = text_to_audio(full_response, lang=audio_lang)
                            if ab:
                                st.audio(ab, format="audio/mp3")
                    else:
                        st.caption("Install gtts for audio")

                # ── Suggested follow-ups ─────
                with st.spinner("💡 Generating related questions…"):
                    sqs = suggested_questions(llm, question, full_response)
                    st.session_state.suggested_questions = sqs

                # ── Persist ──────────────────
                full_with_src = (
                    full_response
                    + "\n\n---\n### 📚 References\n"
                    + "\n".join(src_labels)
                )
                save_conversation(
                    st.session_state.user_id, st.session_state.session_id,
                    "assistant", full_response,
                )
                st.session_state.messages.append(
                    {"role": "user", "content": question}
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_with_src,
                     "sources": src_labels}
                )

    # ═══════════════════════════════════════════
    # TAB 2 — COMPARE SCHEMES
    # ═══════════════════════════════════════════
    with tab_compare:
        st.subheader("📊 Scheme Comparison Tool")

        ALL_SCHEMES = [
            "PM-KISAN", "PMFBY", "Kisan Credit Card (KCC)", "eNAM",
            "PMGSY", "MNREGS", "PM Fasal Bima Yojana", "RKVY",
            "Soil Health Card", "Pradhan Mantri Krishi Sinchayee Yojana",
        ]

        cc1, cc2 = st.columns(2)
        with cc1:
            s1 = st.selectbox("First Scheme", ALL_SCHEMES, key="cmp_s1")
        with cc2:
            s2 = st.selectbox("Second Scheme", ALL_SCHEMES, index=1, key="cmp_s2")

        if st.button("🔄 Compare Now", use_container_width=True):
            if s1 == s2:
                st.warning("Please select two different schemes.")
            else:
                with st.spinner(f"Comparing {s1} vs {s2}…"):
                    res = vectordb.similarity_search_with_score(f"{s1} {s2}", k=6)
                    ctx = "\n\n".join(r[0].page_content for r in res)

                    cmp_prompt = (
                        f"Compare **{s1}** and **{s2}** using the context below.\n"
                        f"Create a detailed markdown table with rows for:\n"
                        f"Objective | Target Beneficiaries | Financial Benefit | "
                        f"Eligibility | Application Process | Ministry | Key Features\n\n"
                        f"Context:\n{ctx}\n\n"
                        f"Return ONLY the markdown table. Write 'Not Available' if info is missing.\n\n"
                        f"| Feature | {s1} | {s2} |\n|---|---|---|"
                    )
                    cmp_resp = llm.invoke(cmp_prompt)
                    st.markdown(cmp_resp.content)

                    if HAS_FPDF:
                        pdf_bytes = generate_pdf(
                            f"Comparison: {s1} vs {s2}",
                            cmp_resp.content, [],
                            st.session_state.username,
                        )
                        if pdf_bytes:
                            st.download_button(
                                "⬇️ Download Comparison PDF",
                                data=pdf_bytes,
                                file_name=f"compare_{s1}_vs_{s2}.pdf",
                                mime="application/pdf",
                            )

    # ═══════════════════════════════════════════
    # TAB 3 — SUMMARISE PDF
    # ═══════════════════════════════════════════
    with tab_summarise:
        st.subheader("📄 Summarise a PDF Document")

        sum_file = st.file_uploader(
            "Upload a scheme PDF to summarise", type=["pdf"], key="sum_pdf"
        )
        if sum_file:
            if st.button("📋 Generate Summary", use_container_width=True):
                os.makedirs("data", exist_ok=True)
                spath = os.path.join("data", sum_file.name)
                with open(spath, "wb") as f:
                    f.write(sum_file.getbuffer())

                with st.spinner("Reading and summarising…"):
                    loader = PyPDFLoader(spath)
                    pages  = loader.load()
                    raw    = "\n".join(p.page_content for p in pages[:5])
                    sum_prompt = (
                        "Summarise this Indian government rural/agricultural scheme document.\n"
                        "Structure your summary with EXACTLY these 5 sections using ## headings:\n"
                        "## 1. Scheme Name & Overview\n"
                        "## 2. Objectives\n"
                        "## 3. Eligibility Criteria\n"
                        "## 4. Key Benefits\n"
                        "## 5. Application Process\n\n"
                        f"Document (first 5 pages):\n{raw[:4000]}"
                    )
                    sum_resp = llm.invoke(sum_prompt)
                    st.markdown(sum_resp.content)

                    # Option to also ingest
                    if st.checkbox("Also add this PDF to the knowledge base", key="sum_ingest"):
                        with st.spinner("Ingesting…"):
                            n = auto_ingest(spath, vectordb)
                            st.success(f"✅ Added {n} chunks to knowledge base.")

                    if HAS_FPDF:
                        pdf_bytes = generate_pdf(
                            f"Summary: {sum_file.name}",
                            sum_resp.content, [],
                            st.session_state.username,
                        )
                        if pdf_bytes:
                            st.download_button(
                                "⬇️ Download Summary PDF",
                                data=pdf_bytes,
                                file_name=f"summary_{sum_file.name}.pdf",
                                mime="application/pdf",
                            )

    # ═══════════════════════════════════════════
    # TAB 4 — ANALYTICS  (all users see summary; officers see full)
    # ═══════════════════════════════════════════
    with tab_analytics:
        if st.session_state.role != "officer":
            st.info("🔒 Full analytics is available for Officer accounts only. Showing your session stats.")
            st.metric("Questions this session", len(st.session_state.questions))
        else:
            st.subheader("📈 Analytics Dashboard")
            stats = get_analytics()

            # Metric row
            m1, m2, m3, m4 = st.columns(4)
            for col, num, lbl in [
                (m1, stats["total_users"],     "👥 Total Users"),
                (m2, stats["total_questions"], "❓ Total Questions"),
                (m3, stats["helpful"],         "👍 Helpful"),
                (m4, stats["not_helpful"],     "👎 Not Helpful"),
            ]:
                with col:
                    st.markdown(
                        f'<div class="metric-box">'
                        f'<div class="num">{num}</div>'
                        f'<div class="lbl">{lbl}</div></div>',
                        unsafe_allow_html=True,
                    )

            st.divider()

            left, right = st.columns(2)

            with left:
                st.subheader("🔝 Top Questions")
                if stats["top_questions"]:
                    for q, cnt in stats["top_questions"][:6]:
                        st.markdown(
                            f"- **{cnt}×** {q[:75]}{'…' if len(q)>75 else ''}"
                        )
                else:
                    st.info("No questions logged yet.")

            with right:
                st.subheader("🌐 Questions by Language")
                if stats["by_language"]:
                    total = max(stats["total_questions"], 1)
                    for lang_code, cnt in stats["by_language"]:
                        name = "🇮🇳 Tamil" if lang_code == "ta" else "🇬🇧 English"
                        st.progress(cnt / total, text=f"{name}: {cnt}")
                else:
                    st.info("No language data yet.")

            st.divider()
            st.subheader("📅 Daily Activity — Last 7 Days")
            if stats["daily_activity"]:
                chart_data = {row[0]: row[1] for row in stats["daily_activity"]}
                st.bar_chart(chart_data)
            else:
                st.info("No activity in the last 7 days.")

            if stats["helpful"] + stats["not_helpful"] > 0:
                st.divider()
                st.subheader("😊 Feedback Breakdown")
                total_fb = stats["helpful"] + stats["not_helpful"]
                h_pct    = round(stats["helpful"] / total_fb * 100)
                st.progress(h_pct / 100, text=f"Helpful: {h_pct}%")


# ──────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────
if st.session_state.logged_in:
    _main_app()
else:
    _login_page()