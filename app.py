import sqlite3
import traceback
import uuid
from datetime import date
from pathlib import Path

import streamlit as st
from langgraph.types import Command

from research_agent.graph.build import build_graph

MAX_RUNS_PER_DAY = 20
USAGE_DB = Path("data/usage.sqlite")
GITHUB_URL = "https://github.com/your-username/Company-Research-Agent"  # change to your repo
DISCLAIMER = "This is for information only and is not investment advice."

EXAMPLES = [
    "Compare TCS and Infosys net margin in FY2025-26",
    "Who is the CEO of Wipro?",
    "What key risks does Infosys name?",
]


@st.cache_resource
def get_graph():
    """Build the graph (and its SQLite checkpointer) once, not on every rerun."""
    return build_graph()


def run_config():
    return {"configurable": {"thread_id": st.session_state.thread_id}, "recursion_limit": 30}


def usage_db():
    """Open the usage database, creating the folder and table on first use."""
    USAGE_DB.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(USAGE_DB)
    conn.execute("CREATE TABLE IF NOT EXISTS usage (date TEXT PRIMARY KEY, count INTEGER NOT NULL)")
    return conn


def runs_today():
    with usage_db() as conn:
        row = conn.execute("SELECT count FROM usage WHERE date = ?", (date.today().isoformat(),)).fetchone()
    return row[0] if row else 0


def count_run():
    with usage_db() as conn:
        conn.execute(
            "INSERT INTO usage (date, count) VALUES (?, 1) "
            "ON CONFLICT(date) DO UPDATE SET count = count + 1",
            (date.today().isoformat(),),
        )


def safe_invoke(payload):
    """Run graph.invoke; on any exception show a short error and return False."""
    try:
        graph.invoke(payload, run_config())
        return True
    except Exception as e:
        traceback.print_exc()  # full traceback goes to the terminal only
        st.error("Something went wrong while researching. Please try again.")
        st.caption(f"Error type: {type(e).__name__}")
        return False


def set_question(text):
    st.session_state["question"] = text


def footer():
    st.divider()
    st.caption(DISCLAIMER)


# set_page_config must be the first Streamlit command
st.set_page_config(page_title="Company Research Agent")
graph = get_graph()

# On page load, pick up a run from the URL (?thread=...) if there is one
if "thread_id" not in st.session_state:
    st.session_state.thread_id = st.query_params.get("thread")

# 1. Header
st.title("Company Research Agent")
st.write(
    "Research on TCS, Infosys and Wipro from financials and FY2025-26 "
    "annual reports, with citations."
)
st.markdown(f"[GitHub repo]({GITHUB_URL})")

# 2. Question input
question = st.text_area("Your question", key="question", height=100)
cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES):
    col.button(example, on_click=set_question, args=(example,), use_container_width=True)

# The research runs ONLY on the rerun caused by clicking Run
if st.button("Run", type="primary"):
    if not question.strip():
        st.error("Please type a question first.")
    elif len(question) > 500:
        st.error("Please keep the question under 500 characters.")
    elif runs_today() >= MAX_RUNS_PER_DAY:
        st.warning("The daily limit is reached. Please try again tomorrow.")
    else:
        count_run()
        st.session_state.thread_id = str(uuid.uuid4())
        st.query_params["thread"] = st.session_state.thread_id  # so a refresh can find this run
        with st.spinner("Researching... this takes about 30 seconds"):
            safe_invoke({"question": question.strip(), "evidence": []})

# Nothing to show until a question has been run
if not st.session_state.thread_id:
    st.info("Type a question, or click an example, then click Run.")
    footer()
    st.stop()

# 3. Read the current state from the checkpointer (survives reruns and restarts)
snapshot = graph.get_state(run_config())
state = snapshot.values
paused = bool(snapshot.next)  # True while waiting at human_review

# 4. Plan
if state.get("plan"):
    st.header("Plan")
    for i, step in enumerate(state["plan"], start=1):
        st.markdown(f"{i}. {step}")

# 5. Evidence (without the long raw content)
if state.get("evidence"):
    st.header("Evidence")
    st.table([
        {"step": e["step"], "tool": e["tool"], "source": e["citation"]}
        for e in state["evidence"]
    ])

# 6. Draft, shown only while the graph waits for approval
if paused:
    st.header("Draft: waiting for your approval")
    st.markdown(state.get("draft", ""))
    for problem in state.get("critique", []):
        st.warning(f"Critic: {problem}")

    if st.button("Approve"):
        with st.spinner("Finishing..."):
            ok = safe_invoke(Command(resume={"action": "approve"}))
        if ok:
            st.rerun()

    feedback = st.text_area("Request changes", key="feedback", height=80)
    if st.button("Send"):
        if not feedback.strip():
            st.error("Type the changes you want first.")
        else:
            with st.spinner("Revising... this takes about 20 seconds"):
                ok = safe_invoke(
                    Command(resume={"action": "revise", "feedback": feedback.strip()})
                )
            if ok:
                st.rerun()

# 7. Final report
if state.get("final_report"):
    st.header("Final report")
    st.markdown(state["final_report"])
    st.download_button(
        "Download report (.md)",
        data=state["final_report"],
        file_name="report.md",
        mime="text/markdown",
    )

footer()