import streamlit as st

GITHUB_URL = "https://github.com/your-username/Company-Research-Agent"  # placeholder

EXAMPLES = [
    "Compare TCS and Infosys net margin in FY2025-26",
    "Who is the CEO of Wipro?",
    "What key risks does Infosys name?",
]

DUMMY_PLAN = [
    "Get TCS and Infosys financials from yfinance",
    "Calculate net margin for each company",
    "Search the annual reports for the reported margin",
]
DUMMY_EVIDENCE = [
    {"fact": "TCS net income: dummy value", "source": "yfinance", "step": 1},
    {"fact": "Infosys net income: dummy value", "source": "yfinance", "step": 1},
    {"fact": "Dummy passage about margin", "source": "TCS annual report, p. 70", "step": 3},
]
DUMMY_DRAFT = "This is a dummy draft. The real draft will come from the writer node."
DUMMY_FINAL = "# Dummy report\n\nThis is a dummy final report with citations."


def set_question(text):
    st.session_state["question"] = text


st.set_page_config(page_title="Company Research Agent")

# 1. Header
st.title("Company Research Agent")
st.write(
    "Research on TCS, Infosys and Wipro from financials and FY2025-26 "
    "annual reports, with citations."
)
st.markdown(f"[GitHub repo]({GITHUB_URL})")

# 2. Question input
st.text_area("Your question", key="question", height=100)
cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES):
    col.button(example, on_click=set_question, args=(example,), use_container_width=True)
run_clicked = st.button("Run", type="primary")

# 3. Status area
status = st.empty()

# 4. Placeholder sections
st.header("Plan")
for i, step in enumerate(DUMMY_PLAN, start=1):
    st.markdown(f"{i}. {step}")

st.header("Evidence")
st.table(DUMMY_EVIDENCE)

st.header("Draft")
st.write(DUMMY_DRAFT)
st.button("Approve")
st.text_area("Request changes", key="feedback", height=80)
st.button("Send")

st.header("Final report")
st.write(DUMMY_FINAL)
st.download_button(
    "Download report (.md)",
    data=DUMMY_FINAL,
    file_name="report.md",
    mime="text/markdown",
)

# 5. Footer
st.divider()
st.caption("This is for information only and is not investment advice.")
