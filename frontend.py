"""
app.py — Streamlit frontend for the text-to-SQL clarification pipeline.

Assumes your LangGraph pipeline file exposes a compiled `workflow` object.
Adjust the import line below if your file isn't named `graph.py`.
"""

import uuid
import streamlit as st
from langgraph.types import Command
from agent import workflow  # <-- change "agent" if your pipeline file has a different name

st.set_page_config(page_title="Ledger — Ask your database", page_icon="🗄️", layout="centered")

# ---------------------------------------------------------------------------
# Design tokens (kept in one place so the palette/type system stays consistent)
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg: #0B1120;
    --surface: #131C31;
    --surface-raised: #1B2740;
    --border: #263250;
    --text: #E8EDF6;
    --text-muted: #8792A6;
    --accent: #4FD1C5;
    --accent-dim: #2C6E68;
    --amber: #F5A623;
}

.stApp { background-color: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { background-color: var(--surface); border-right: 1px solid var(--border); }
[data-testid="stChatInput"] textarea { font-family: 'JetBrains Mono', monospace; }

h1, h2, h3 { font-family: 'JetBrains Mono', monospace; letter-spacing: -0.02em; }

.app-header {
    display: flex; align-items: baseline; gap: 12px; margin-bottom: 4px;
}
.app-header .mark {
    font-family: 'JetBrains Mono', monospace; font-size: 26px; font-weight: 700;
    color: var(--accent); letter-spacing: -0.03em;
}
.app-tagline { color: var(--text-muted); font-size: 14px; margin-bottom: 28px; }

/* --- pipeline stage tracker: the signature element --- */
.trail { display: flex; align-items: center; gap: 0; margin: 8px 0 28px 0; }
.trail-step { display: flex; align-items: center; }
.trail-dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: var(--border); border: 2px solid var(--border);
    transition: all 0.3s ease;
}
.trail-dot.done { background: var(--accent); border-color: var(--accent); }
.trail-dot.active {
    background: var(--amber); border-color: var(--amber);
    box-shadow: 0 0 0 4px rgba(245, 166, 35, 0.2);
}
.trail-label {
    font-family: 'JetBrains Mono', monospace; font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--text-muted); margin-left: 6px; margin-right: 10px;
}
.trail-label.done { color: var(--accent); }
.trail-label.active { color: var(--amber); }
.trail-line { flex: 1; height: 1px; background: var(--border); min-width: 16px; margin-right: 10px; }
.trail-line.done { background: var(--accent-dim); }

/* clarification callout */
.clarify-box {
    background: rgba(245, 166, 35, 0.08); border: 1px solid rgba(245, 166, 35, 0.35);
    border-radius: 8px; padding: 14px 16px; margin: 8px 0;
    font-family: 'Inter', sans-serif;
}
.clarify-box .eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--amber); margin-bottom: 6px;
}

/* sql + result panels */
.panel-label {
    font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--text-muted); margin: 18px 0 6px 0;
}

/* example prompt chips */
.stButton > button {
    background: var(--surface-raised); border: 1px solid var(--border);
    color: var(--text); font-family: 'Inter', sans-serif; font-size: 13px;
    text-align: left; border-radius: 8px; padding: 10px 14px;
}
.stButton > button:hover { border-color: var(--accent); color: var(--accent); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Pipeline stage tracker
# ---------------------------------------------------------------------------
STAGES = [
    ("schemaLinker", "Linking schema"),
    ("ambiguityAgent", "Checking clarity"),
    ("sqlQuery", "Generating SQL"),
    ("executeAgent", "Running query"),
]


def render_trail(current_stage: str | None, waiting_on_user: bool, all_done: bool = False):
    html = '<div class="trail">'
    for i, (key, label) in enumerate(STAGES):
        if all_done:
            dot_class, label_class = "done", "done"
        elif waiting_on_user and key == "ambiguityAgent":
            dot_class, label_class = "active", "active"
        elif current_stage is None:
            dot_class, label_class = "", ""
        else:
            stage_order = [s[0] for s in STAGES]
            done = stage_order.index(key) < stage_order.index(current_stage) if current_stage in stage_order else False
            is_current = key == current_stage
            dot_class = "done" if done else ("active" if is_current else "")
            label_class = dot_class
        html += f'<div class="trail-step"><div class="trail-dot {dot_class}"></div>'
        html += f'<div class="trail-label {label_class}">{label}</div></div>'
        if i < len(STAGES) - 1:
            line_class = "done" if dot_class == "done" else ""
            html += f'<div class="trail-line {line_class}"></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ...}
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "current_stage" not in st.session_state:
    st.session_state.current_stage = None
if "run_complete" not in st.session_state:
    st.session_state.run_complete = False
if "pending_pipeline_input" not in st.session_state:
    st.session_state.pending_pipeline_input = None

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🗄️ Ledger")
    st.caption("Text-to-SQL with a clarification loop")
    st.divider()
    if st.button("🔁 New conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.session_state.current_stage = None
        st.session_state.run_complete = False
        st.session_state.pending_pipeline_input = None
        st.rerun()
    st.divider()
    st.caption(
        "Ambiguous questions get one clarifying question back before a query "
        "runs. All generated SQL is read-only."
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="app-header"><span class="mark">&gt; Text to SQL with clarification engine</span></div>', unsafe_allow_html=True)
st.markdown('<div class="app-tagline">Ask your database a question in plain English.</div>', unsafe_allow_html=True)

trail_slot = st.empty()
with trail_slot:
    render_trail(
        st.session_state.current_stage,
        st.session_state.pending_question is not None,
        all_done=st.session_state.run_complete,
    )


EXAMPLE_PROMPTS = [
    "Show me the top 5 customers by total spending",
    "Which products are out of stock?",
    "How many orders were placed last month?",
]

if not st.session_state.messages:
    st.markdown(
        '<div class="panel-label" style="margin-top:8px;">Try asking</div>',
        unsafe_allow_html=True,
    )
    example_cols = st.columns(len(EXAMPLE_PROMPTS))
    for col, prompt_text in zip(example_cols, EXAMPLE_PROMPTS):
        with col:
            if st.button(prompt_text, use_container_width=True, key=f"ex_{prompt_text}"):
                st.session_state.messages.append({"role": "user", "content": prompt_text})
                st.session_state.pending_pipeline_input = prompt_text
                st.rerun()

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🗄️"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg.get("type") == "clarify":
            st.markdown(
                f'<div class="clarify-box"><div class="eyebrow">Needs clarification</div>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        elif msg.get("type") == "result":
            if msg.get("sql"):
                st.markdown('<div class="panel-label">Generated SQL</div>', unsafe_allow_html=True)
                st.code(msg["sql"], language="sql")
                st.markdown('<div class="panel-label">Result</div>', unsafe_allow_html=True)
            st.markdown(msg["content"])
        else:
            st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Run the graph, updating the stage tracker live via .stream()
# ---------------------------------------------------------------------------
def run_pipeline(payload):
    final_state = None
    for step in workflow.stream(payload, config, stream_mode="updates"):
        if "__interrupt__" in step:
            return {"__interrupt__": step["__interrupt__"]}
        node_name = next(iter(step))
        st.session_state.current_stage = node_name
        with trail_slot:
            render_trail(st.session_state.current_stage, False)
        final_state = workflow.get_state(config).values
    return final_state


# ---------------------------------------------------------------------------
# Process any pipeline input queued by the previous run (see chat_input
# section below). Doing this on a dedicated rerun — rather than inline in
# the same run as the submission — avoids chat_input occasionally "eating"
# a submission during a slow (multi-second LLM) run.
# ---------------------------------------------------------------------------
if st.session_state.pending_pipeline_input is not None:
    to_process = st.session_state.pending_pipeline_input
    st.session_state.pending_pipeline_input = None

    with st.chat_message("assistant", avatar="🗄️"):
        with st.spinner("Working..."):
            try:
                if st.session_state.pending_question:
                    outcome = run_pipeline(Command(resume=to_process))
                else:
                    st.session_state.current_stage = "schemaLinker"
                    st.session_state.run_complete = False
                    with trail_slot:
                        render_trail("schemaLinker", False)
                    outcome = run_pipeline({"user": to_process})
            except RuntimeError as e:
                st.error(str(e))
                st.session_state.messages.append({"role": "assistant", "content": str(e)})
                outcome = None

        if outcome and "__interrupt__" in outcome:
            question = outcome["__interrupt__"][0].value["question"]
            st.session_state.pending_question = question
            st.markdown(
                f'<div class="clarify-box"><div class="eyebrow">Needs clarification</div>{question}</div>',
                unsafe_allow_html=True,
            )
            st.session_state.messages.append({"role": "assistant", "type": "clarify", "content": question})
            with trail_slot:
                render_trail("ambiguityAgent", True)
        elif outcome:
            st.session_state.pending_question = None
            sql = outcome.get("query", "")
            result = outcome.get("result", "No result.")
            if sql:
                st.markdown('<div class="panel-label">Generated SQL</div>', unsafe_allow_html=True)
                st.code(sql, language="sql")
                st.markdown('<div class="panel-label">Result</div>', unsafe_allow_html=True)
            st.markdown(result)
            st.session_state.messages.append(
                {"role": "assistant", "type": "result", "sql": sql, "content": result}
            )
            st.session_state.current_stage = None
            st.session_state.run_complete = True
            with trail_slot:
                render_trail(None, False, all_done=True)

    st.rerun()

# ---------------------------------------------------------------------------
# Chat input — only queues the submission and reruns; processing happens
# above, on the next run, not inline here.
# ---------------------------------------------------------------------------
placeholder = "Type your answer..." if st.session_state.pending_question else "Ask a question about your data..."
user_input = st.chat_input(placeholder)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.pending_pipeline_input = user_input
    st.rerun()