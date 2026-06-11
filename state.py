from typing import TypedDict, Annotated
import operator


class ResearchState(TypedDict):
    query: str
    search_results: list[dict]
    scraped_content: list[str]
    analysis: str
    final_report: str
    messages: Annotated[list, operator.add]
