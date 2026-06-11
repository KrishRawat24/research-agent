import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Multi-Agent Research Assistant")
st.caption("Powered by LangGraph · GPT-4o-mini · Tavily Search")

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("How it works")
    st.markdown("""
1. **Orchestrator** — plans the research strategy
2. **Search agent** — queries Tavily for live web results
3. **Scraper agent** — fetches full content from top URLs
4. **Analyst agent** — extracts key findings & gaps
5. **Writer agent** — produces a structured Markdown report
    """)
    st.divider()
    st.markdown("Built with [LangGraph](https://github.com/langchain-ai/langgraph) · [FastAPI](https://fastapi.tiangolo.com) · [Streamlit](https://streamlit.io)")

# ─── Main ────────────────────────────────────────────────────────────────────
query = st.text_input(
    "What do you want to research?",
    placeholder="e.g. Latest developments in agentic AI frameworks 2026",
)

col1, col2 = st.columns([1, 5])
with col1:
    run = st.button("🚀 Research", type="primary", use_container_width=True)

if run and query:
    with st.spinner("Agents working... (usually 20-40 seconds)"):
        try:
            resp = requests.post(
                f"{API_URL}/research",
                json={"query": query},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

            st.success("Research complete!")

            tab1, tab2 = st.tabs(["📄 Report", "🔬 Raw Analysis"])

            with tab1:
                st.markdown(data["report"])
                st.download_button(
                    "⬇️ Download report",
                    data["report"],
                    file_name="research_report.md",
                    mime="text/markdown",
                )

            with tab2:
                st.text_area("Analysis", data["analysis"], height=400)

            st.caption(f"Thread ID: `{data['thread_id']}`")

        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the API. Make sure it's running at: " + API_URL)
        except Exception as e:
            st.error(f"Error: {e}")

elif run and not query:
    st.warning("Please enter a research query first.")
