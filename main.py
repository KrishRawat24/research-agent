import os
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

from src.graph import compiled_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup checks
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError("TAVILY_API_KEY is not set")
    yield


app = FastAPI(
    title="Multi-Agent Research Assistant",
    description="LangGraph-powered research pipeline with 4 autonomous agents",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ─────────────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str
    thread_id: str | None = None  # Pass same thread_id to continue a session


class ResearchResponse(BaseModel):
    thread_id: str
    query: str
    report: str
    analysis: str


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": os.getenv("LLM_MODEL", "gpt-4o-mini")}


@app.post("/research", response_model=ResearchResponse)
async def research(req: ResearchRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = await compiled_graph.ainvoke(
            {"query": req.query, "messages": [], "search_results": [],
             "scraped_content": [], "analysis": "", "final_report": ""},
            config=config,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ResearchResponse(
        thread_id=thread_id,
        query=req.query,
        report=result["final_report"],
        analysis=result["analysis"],
    )
