import os
import httpx
from bs4 import BeautifulSoup
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

# Tavily search — returns top 5 results with URLs + snippets
search_tool = TavilySearchResults(max_results=5)


@tool
def scrape_url(url: str) -> str:
    """Scrapes and returns clean text content from a URL. Truncated to 4000 chars."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (research-agent/1.0)"}
        resp = httpx.get(url, timeout=10, headers=headers, follow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove script/style noise
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text[:4000]
    except Exception as e:
        return f"Failed to scrape {url}: {str(e)}"
