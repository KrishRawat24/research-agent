from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.state import ResearchState
from src.agents import (
    orchestrator_node,
    search_node,
    scraper_node,
    analyst_node,
    writer_node,
)


def build_graph():
    graph = StateGraph(ResearchState)

    # Register nodes
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("search",       search_node)
    graph.add_node("scraper",      scraper_node)
    graph.add_node("analyst",      analyst_node)
    graph.add_node("writer",       writer_node)

    # Flow
    graph.set_entry_point("orchestrator")
    graph.add_edge("orchestrator", "search")
    graph.add_edge("search",       "scraper")
    graph.add_edge("scraper",      "analyst")
    graph.add_edge("analyst",      "writer")
    graph.add_edge("writer",       END)

    # In-memory checkpointing (swap for SqliteSaver in production)
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# Singleton — imported by the API
compiled_graph = build_graph()
