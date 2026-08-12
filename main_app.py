import streamlit as st
import time
import re
import threading
import uuid
import pandas as pd

from src.pdf_processor import (
    load_pdf_text,
    parse_inbody,
    create_chunks
)

from src.vector_store import VectorStore
from src.llm import load_llm
from src.rag_pipeline import RAGPipeline


# ============================================================
# CHAT SESSION HELPERS
# ============================================================

# In main_app.py, update the streaming function:

def _generate_answer_streaming(pipeline, question, result_container):
    """Generates answer with streaming tokens - updates in real-time as tokens are generated."""
    try:
        full_answer = ""
        for chunk in pipeline.answer_stream(question):
            # Update with each new chunk as it arrives
            full_answer = chunk
            result_container["partial_answer"] = chunk
            
            
        
        result_container["answer"] = full_answer
    except Exception as e:
        result_container["error"] = str(e)
    finally:
        result_container["done"] = True


def _build_pipeline_history(ui_messages):
    """Converts the UI's [{'role','content'}, ...] transcript into the
    [{'user','assistant'}, ...] shape RAGPipeline.chat_history expects."""

    history = []
    pending_user = None

    for msg in ui_messages:

        if msg["role"] == "user":
            pending_user = msg["content"]

        elif msg["role"] == "assistant" and pending_user is not None:
            history.append({
                "user": pending_user,
                "assistant": msg["content"]
            })
            pending_user = None

    return history


def _archive_current_chat():
    """Saves the in-progress conversation into chat_sessions (creating
    a new entry, or updating the one currently active) before it gets
    cleared or swapped out."""

    if not st.session_state.chat_messages:
        return

    title = "Chat"

    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            content = msg["content"]
            title = content[:40] + ("…" if len(content) > 40 else "")
            break

    session_id = st.session_state.active_session_id or str(uuid.uuid4())
    pipeline_history = _build_pipeline_history(st.session_state.chat_messages)

    existing = next(
        (s for s in st.session_state.chat_sessions if s["id"] == session_id),
        None
    )

    if existing:
        existing["messages"] = list(st.session_state.chat_messages)
        existing["pipeline_history"] = pipeline_history

    else:
        st.session_state.chat_sessions.insert(0, {
            "id": session_id,
            "title": title,
            "messages": list(st.session_state.chat_messages),
            "pipeline_history": pipeline_history,
            "timestamp": time.strftime("%b %d, %I:%M %p")
        })


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NutriMind AI",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(56, 189, 248, 0.10),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(34, 197, 94, 0.08),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #07111f 0%,
            #0b1726 45%,
            #07131d 100%
        );
}

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #07111f 0%,
            #0b1726 100%
        );

    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {
    color: #e5f2ff;
}

.hero {
    padding: 35px 40px;
    border-radius: 24px;

    background:
        linear-gradient(
            135deg,
            rgba(15, 118, 110, 0.35),
            rgba(14, 165, 233, 0.16)
        );

    border: 1px solid rgba(255,255,255,0.10);

    box-shadow:
        0 20px 60px rgba(0,0,0,0.25);

    margin-bottom: 25px;

    animation: fadeIn 0.8s ease-out;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    color: white;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 16px;
    color: #a9c4d9;
    max-width: 750px;
}

.logo {
    font-size: 24px;
    font-weight: 800;
    color: white;
}

.metric-card {

    padding: 22px;

    border-radius: 18px;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.07),
            rgba(255,255,255,0.025)
        );

    border:
        1px solid rgba(255,255,255,0.08);

    box-shadow:
        0 12px 35px rgba(0,0,0,0.20);

    transition:
        transform 0.25s ease,
        border 0.25s ease;

    animation: slideUp 0.6s ease-out;
}

.metric-card:hover {

    transform: translateY(-5px);

    border:
        1px solid rgba(56,189,248,0.45);

}

.metric-label {

    color: #8da8bb;

    font-size: 13px;

    font-weight: 600;

    text-transform: uppercase;

    letter-spacing: 0.08em;

}

.metric-value {

    color: white;

    font-size: 30px;

    font-weight: 800;

    margin-top: 6px;

}

.metric-icon {

    font-size: 24px;

    margin-bottom: 8px;

}

.section-title {

    color: white;

    font-size: 23px;

    font-weight: 750;

    margin-top: 25px;

    margin-bottom: 15px;

}

.upload-box {

    padding: 25px;

    border-radius: 20px;

    border: 1px dashed rgba(56,189,248,0.45);

    background:
        rgba(14,165,233,0.05);

}

.chat-user {

    background:
        linear-gradient(
            135deg,
            #0ea5e9,
            #0284c7
        );

    color: white;

    padding: 13px 17px;

    border-radius:
        18px 18px 4px 18px;

    margin:
        8px 0 8px auto;

    max-width: 75%;

}

.chat-ai {

    background:
        rgba(255,255,255,0.06);

    border:
        1px solid rgba(255,255,255,0.08);

    color: #dbeafe;

    padding: 14px 17px;

    border-radius:
        18px 18px 18px 4px;

    margin:
        8px auto 8px 0;

    max-width: 80%;

}

.progress-container {

    height: 10px;

    border-radius: 20px;

    background:
        rgba(255,255,255,0.08);

    overflow: hidden;

}

.progress-fill {

    height: 100%;

    border-radius: 20px;

    background:
        linear-gradient(
            90deg,
            #06b6d4,
            #22c55e
        );

    animation:
        progressAnimation 1.2s ease-out;

}

.status {

    padding: 10px 15px;

    border-radius: 12px;

    background:
        rgba(34,197,94,0.10);

    border:
        1px solid rgba(34,197,94,0.25);

    color: #86efac;

    font-size: 13px;

}

@keyframes fadeIn {

    from {
        opacity: 0;
        transform: translateY(-10px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }

}

@keyframes slideUp {

    from {
        opacity: 0;
        transform: translateY(20px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }

}

@keyframes progressAnimation {

    from {
        width: 0;
    }

}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "inbody_data" not in st.session_state:
    st.session_state.inbody_data = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []

if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None

if "generating" not in st.session_state:
    st.session_state.generating = False

if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False

if "answer_result" not in st.session_state:
    st.session_state.answer_result = {}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="logo">🏋️ NutriMind AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='color:#8da8bb;'>Your intelligent InBody fitness coach</p>",
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### 📄 Your InBody Report")

    uploaded_file = st.file_uploader(
        "Upload your InBody PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )

    st.divider()

    if st.session_state.analysis_complete:

        st.markdown(
            """<div class="status">
✓ InBody report analyzed
</div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.markdown("### 💬 Conversation")

    if st.session_state.pipeline is None:

        st.caption("Upload a report above to start chatting.")

    else:

        col_new, col_stop = st.columns(2)

        with col_new:

            if st.button(
                "🆕 New Chat",
                disabled=st.session_state.generating,
                use_container_width=True
            ):

                _archive_current_chat()

                st.session_state.chat_messages = []
                st.session_state.active_session_id = None
                st.session_state.pipeline.clear_memory()

                st.rerun()

        with col_stop:

            if st.button(
                "🛑 Stop Chat",
                disabled=not st.session_state.generating,
                use_container_width=True
            ):

                st.session_state.stop_requested = True
                st.session_state.generating = False

                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": "⏹️ Response stopped."
                })

                st.rerun()

        st.markdown("#### 🕑 Chat History")

        if not st.session_state.chat_sessions:

            st.caption("No past chats yet.")

        else:

            for session in st.session_state.chat_sessions:

                is_active = session["id"] == st.session_state.active_session_id

                label = ("📍 " if is_active else "") + session["title"]

                if st.button(
                    label,
                    key=f"history_{session['id']}",
                    disabled=st.session_state.generating,
                    use_container_width=True,
                    help=session["timestamp"]
                ):

                    _archive_current_chat()

                    st.session_state.chat_messages = list(session["messages"])
                    st.session_state.active_session_id = session["id"]
                    st.session_state.pipeline.chat_history = list(
                        session["pipeline_history"]
                    )

                    st.rerun()

    st.markdown("---")

    st.caption(
        "NutriMind AI provides general informational guidance "
        "and is not a replacement for professional medical advice."
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """<div class="hero">

<div class="hero-title">
Your Body. Your Data. Your Plan. 💪
</div>

<div class="hero-subtitle">
Upload your InBody report and let NutriMind AI
turn your body-composition data into personalized
fitness and nutrition insights.
</div>

</div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    if (
        st.session_state.pdf_name != uploaded_file.name
        or st.session_state.pipeline is None
    ):

        with st.spinner("🔍 Analyzing your InBody report..."):

            pdf_path = "data/current_inbody.pdf"

            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            raw_text = load_pdf_text(pdf_path)

            inbody_data = parse_inbody(raw_text)

            chunks = create_chunks(inbody_data)

            vector_store = VectorStore(
                chunks=chunks,
                path="data/vector_store",
                force_rebuild=True
            )

            tokenizer, model = load_llm()

            pipeline = RAGPipeline(
                vector_store,
                tokenizer,
                model,
                inbody_data=inbody_data
            )

            st.session_state.inbody_data = inbody_data
            st.session_state.pipeline = pipeline
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.chat_messages = []
            st.session_state.analysis_complete = True

        st.success("Your InBody report has been analyzed successfully! 🎯")


# ============================================================
# DASHBOARD
# ============================================================

if st.session_state.inbody_data:

    data = st.session_state.inbody_data

    st.markdown(
        '<div class="section-title">📊 Body Composition Overview</div>',
        unsafe_allow_html=True
    )

    def flatten_data(raw_data):
        rows = []

        for section, entries in raw_data.items():
            for entry in entries:
                if ":" in entry:
                    label, value = entry.split(":", 1)
                    rows.append({
                        "section": section,
                        "label": label.strip(),
                        "value": value.strip()
                    })

        return rows

    flat_rows = flatten_data(data)

    def find_value(possible_labels):

        for row in flat_rows:
            if row["label"] in possible_labels:
                return row["value"]

        return "—"

    def to_float(value):
        match = re.search(r"[-+]?\d*\.?\d+", str(value))
        return float(match.group()) if match else None

    col1, col2, col3, col4 = st.columns(4)

    metrics = [

        (
            col1,
            "⚖️",
            "Weight",
            find_value(["Weight"])
        ),

        (
            col2,
            "📏",
            "BMI",
            find_value(["BMI (Body Mass Index)"])
        ),

        (
            col3,
            "🔥",
            "Body Fat",
            find_value(["Percent Body Fat (PBF)"])
        ),

        (
            col4,
            "💪",
            "Skeletal Muscle",
            find_value(["Skeletal Muscle Mass (SMM)"])
        )

    ]

    for col, icon, label, value in metrics:

        with col:

            st.markdown(
                f"""<div class="metric-card">

<div class="metric-icon">
{icon}
</div>

<div class="metric-label">
{label}
</div>

<div class="metric-value">
{value}
</div>

</div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # BODY COMPOSITION CHART
    # ========================================================

    st.markdown(
        '<div class="section-title">💪 Body Composition</div>',
        unsafe_allow_html=True
    )

    chart_data = {}

    weight = find_value(["Weight"])
    bmi = find_value(["BMI (Body Mass Index)"])
    body_fat = find_value(["Percent Body Fat (PBF)"])
    muscle = find_value(["Skeletal Muscle Mass (SMM)"])

    weight_f = to_float(weight)
    body_fat_f = to_float(body_fat)
    muscle_f = to_float(muscle)

    if weight_f is not None and body_fat_f is not None and muscle_f is not None:

        chart_data = pd.DataFrame({
            "Metric": [
                "Weight",
                "Body Fat %",
                "Skeletal Muscle"
            ],
            "Value": [
                weight_f,
                body_fat_f,
                muscle_f
            ]
        })

        st.bar_chart(
            chart_data.set_index("Metric")
        )

    else:

        st.info(
            "Body composition chart will appear when numerical "
            "values are available in the parsed report."
        )


    # ========================================================
    # TABS
    # ========================================================

    overview_tab, coach_tab, report_tab = st.tabs(
        [
            "🏠 Overview",
            "🤖 AI Coach",
            "📋 Full Report"
        ]
    )


    # ========================================================
    # OVERVIEW TAB
    # ========================================================

    with overview_tab:

        st.markdown(
            '<div class="section-title">🎯 Your Fitness Dashboard</div>',
            unsafe_allow_html=True
        )

        st.write(
            "Your dashboard is generated directly from the "
            "information extracted from your InBody report."
        )

        st.markdown(
            f"""<div class="metric-card">

<h3 style="color:white;">
📌 Your Numbers at a Glance
</h3>

<p style="color:#c7d9e6; line-height:1.7;">
Current weight is <b>{weight}</b>, with a BMI of <b>{bmi}</b>.
Body fat sits at <b>{body_fat}%</b>, and skeletal muscle mass
is <b>{muscle}</b>, based on your most recently uploaded
InBody report.
</p>

</div>
            """,
            unsafe_allow_html=True
        )

        shown_labels = {
            "Weight",
            "BMI (Body Mass Index)",
            "Percent Body Fat (PBF)",
            "Skeletal Muscle Mass (SMM)"
        }

        extra_rows = [
            row for row in flat_rows
            if row["label"] not in shown_labels
        ]

        if extra_rows:

            st.markdown(
                '<div class="section-title" style="font-size:18px;">'
                '📎 Other Values From Your Report</div>',
                unsafe_allow_html=True
            )

            extra_df = pd.DataFrame(
                [
                    {
                        "Section": row["section"].title(),
                        "Metric": row["label"],
                        "Value": row["value"]
                    }
                    for row in extra_rows
                ]
            )

            st.dataframe(
                extra_df,
                use_container_width=True,
                hide_index=True
            )

        st.markdown(
            """<div class="metric-card">

<h3 style="color:white;">
🚀 Ready to work toward your goals?
</h3>

<p style="color:#9fb5c7;">
Ask NutriMind AI about your body composition,
meal planning, or your next fitness step.
</p>

</div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # AI COACH TAB (WITH STREAMING)
    # ========================================================

    with coach_tab:

        st.markdown(
            '<div class="section-title">🤖 Your AI Fitness Coach</div>',
            unsafe_allow_html=True
        )

        st.caption(
            "Ask questions about your InBody report, meals, "
            "or body composition."
        )

        # Display existing chat messages
        for message in st.session_state.chat_messages:

            if message["role"] == "user":

                st.markdown(
                    f"""<div class="chat-user">{message["content"]}</div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""<div class="chat-ai">
🤖 <b>NutriMind AI</b><br><br>
{message["content"]}
</div>
                    """,
                    unsafe_allow_html=True
                )

        # Streaming response handling
        if st.session_state.generating:
            
            result = st.session_state.answer_result
            
            # Show partial answer if available
            if "partial_answer" in result and result["partial_answer"]:
                st.markdown(
                    f"""<div class="chat-ai">
🤖 <b>NutriMind AI</b><br><br>
{result["partial_answer"]}▌
</div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """<div class="chat-ai">
🤖 <b>NutriMind AI</b><br><br>
🤔 Thinking... (use "Stop Chat" in the sidebar to cancel)
</div>
                    """,
                    unsafe_allow_html=True
                )
            
            if result.get("done"):
                st.session_state.generating = False
                
                if not st.session_state.stop_requested:
                    if "error" in result:
                        content = f"⚠️ Something went wrong: {result['error']}"
                    else:
                        content = result.get("answer", "")
                    
                    # Only add if not already added
                    if not st.session_state.chat_messages or st.session_state.chat_messages[-1].get("content") != content:
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": content
                        })
                
                st.session_state.stop_requested = False
                # Clear partial answer
                if "partial_answer" in st.session_state.answer_result:
                    del st.session_state.answer_result["partial_answer"]
                st.rerun()
            
            else:
                time.sleep(0.2)
                st.rerun()

        else:
            question = st.chat_input(
                "Ask your fitness coach anything about your InBody..."
            )
            
            if question:
                st.session_state.chat_messages.append({
                    "role": "user",
                    "content": question
                })
                
                st.session_state.answer_result = {"done": False}
                st.session_state.stop_requested = False
                st.session_state.generating = True
                
                thread = threading.Thread(
                    target=_generate_answer_streaming,
                    args=(
                        st.session_state.pipeline,
                        question,
                        st.session_state.answer_result
                    ),
                    daemon=True
                )
                
                st.session_state.answer_thread = thread
                thread.start()
                
                st.rerun()

        # Clear chat button
        col1, col2 = st.columns([1, 5])

        with col1:

            if st.button(
                "🧹 Clear Chat",
                disabled=st.session_state.generating
            ):

                st.session_state.chat_messages = []
                st.session_state.active_session_id = None

                if st.session_state.pipeline:

                    st.session_state.pipeline.clear_memory()

                st.rerun()


    # ========================================================
    # FULL REPORT TAB
    # ========================================================

    with report_tab:

        st.markdown(
            '<div class="section-title">📋 Extracted InBody Data</div>',
            unsafe_allow_html=True
        )

        if flat_rows:

            report_df = pd.DataFrame(
                [
                    {
                        "Section": row["section"].title(),
                        "Metric": row["label"],
                        "Value": row["value"]
                    }
                    for row in flat_rows
                ]
            )

            st.dataframe(
                report_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.write(data)


else:

    # ========================================================
    # EMPTY STATE
    # ========================================================

    st.markdown(
        """<div style="
text-align:center;
padding:80px 20px;
animation:fadeIn 1s ease-out;
        ">

<div style="
font-size:80px;
margin-bottom:20px;
            ">
🏋️
</div>

<h2 style="color:white;">
Start Your Fitness Journey
</h2>

<p style="
color:#8da8bb;
font-size:16px;
max-width:600px;
margin:auto;
            ">
Upload your InBody PDF from the sidebar.
NutriMind AI will analyze your body composition
and create a personalized fitness and nutrition
dashboard.
            </p>

</div>
        """,
        unsafe_allow_html=True
    )