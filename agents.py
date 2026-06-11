import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from src.state import ResearchState
from src.tools import search_tool, scrape_url

_model = os.getenv("LLM_MODEL", "gpt-4o-mini")


def _llm(temperature: float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(model=_model, temperature=temperature)


# ─── Orchestrator ────────────────────────────────────────────────────────────

def orchestrator_node(state: ResearchState) -> dict:
    """Plans the research strategy and kicks off parallel sub-tasks."""
    llm = _llm()
    response = llm.invoke([
        SystemMessage(content=(
            "You are a research orchestrator. Given a user query, "
            "produce a brief plan (2-3 sentences) describing what to search for "
            "and what aspects to analyse. Be concise."
        )),
        HumanMessage(content=state["query"]),
    ])
    return {
        "messages": [{"role": "orchestrator", "content": response.content}],
    }


# ─── Search Agent ────────────────────────────────────────────────────────────

def search_node(state: ResearchState) -> dict:
    """Runs a web search and returns structured results."""
    results = search_tool.invoke({"query": state["query"]})
    # results is a list of dicts: {url, content, title, score}
    return {"search_results": results}


# ─── Scraper Agent ───────────────────────────────────────────────────────────

def scraper_node(state: ResearchState) -> dict:
    """Scrapes the top 3 URLs from search results for full content."""
    scraped = []
    urls = [r["url"] for r in state.get("search_results", [])[:3]]
    for url in urls:
        text = scrape_url.invoke({"url": url})
        scraped.append(f"[Source: {url}]\n{text}")
    return {"scraped_content": scraped}


# ─── Analyst Agent ───────────────────────────────────────────────────────────

def analyst_node(state: ResearchState) -> dict:
    """Synthesises search results + scraped content into structured analysis."""
    llm = _llm(temperature=0.1)

    search_summary = "\n\n".join(
        f"- {r.get('title', 'No title')}: {r.get('content', '')}"
        for r in state.get("search_results", [])
    )
    scraped_summary = "\n\n---\n\n".join(state.get("scraped_content", []))

    prompt = f"""Query: {state['query']}

Search results:
{search_summary}

Full page content:
{scraped_summary}

Analyse the above. Extract:
1. Key facts and findings
2. Consensus points across sources
3. Contradictions or gaps
4. Most credible sources

Be structured and concise."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"analysis": response.content}


# ─── Writer Agent ────────────────────────────────────────────────────────────

def writer_node(state: ResearchState) -> dict:
    """Converts analysis into a polished Markdown research report."""
    llm = _llm(temperature=0.5)

    prompt = f"""You are a professional research writer.

Query: {state['query']}

Analysis:
{state.get('analysis', '')}

Write a well-structured Markdown research report with:
- An executive summary (2-3 sentences)
- Key findings (bullet points)
- Detailed breakdown (2-3 sections with headers)
- Sources / further reading
- Confidence level (High / Medium / Low) with a one-line justification

Use clear, professional language. No fluff."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_report": response.content}
