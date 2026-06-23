"""
main.py
-------
FastAPI backend for the RAG AI Knowledge Chatbot.

Endpoints:
  POST /ingest  → runs the full ingestion pipeline
  POST /chat    → accepts {"query": "..."}, returns {"answer": "...", "sources": [...]}
  GET  /health  → returns {"status": "ok"}

Run with:
    uvicorn backend.main:app --reload
"""

import os
import sys

# Ensure backend directory is on the path for sibling imports
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Load environment variables
# ─────────────────────────────────────────────
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set. Create a .env file with your key.")

# Import pipeline modules
from ingestion import run_ingestion
from chat import chat as rag_chat

# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(
    title="RAG AI Knowledge Chatbot",
    description=(
        "A Retrieval-Augmented Generation chatbot powered by Gemini and ChromaDB. "
        "Ingest documents, then ask questions grounded in your knowledge base."
    ),
    version="1.0.0",
)

# ─────────────────────────────────────────────
# CORS — allow all origins so the frontend HTML file can call the API
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


class IngestResponse(BaseModel):
    status: str
    message: str
    details: dict = {}


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/health", summary="Health check")
async def health_check():
    """Returns OK if the server is running."""
    return {"status": "ok", "message": "RAG AI Chatbot backend is running."}


@app.post("/ingest", response_model=IngestResponse, summary="Run ingestion pipeline")
async def ingest():
    """
    Triggers the full ingestion pipeline:
      - Load PDF (ai_knowledge.pdf)
      - Scrape blog articles
      - Chunk, embed, and store in ChromaDB
    """
    try:
        result = run_ingestion()

        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message", "Ingestion failed."))

        return IngestResponse(
            status="success",
            message=(
                f"Ingestion complete. "
                f"{result.get('documents_loaded', 0)} documents loaded, "
                f"{result.get('chunks_created', 0)} chunks stored."
            ),
            details=result,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")


@app.post("/chat", response_model=ChatResponse, summary="Chat with the knowledge base")
async def chat_endpoint(request: ChatRequest):
    """
    Accept a user query, retrieve relevant chunks from ChromaDB,
    and return a Gemini-generated answer grounded in those chunks.
    """
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = rag_chat(query)
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
        )
    except RuntimeError as e:
        # Likely: vector store not populated yet
        raise HTTPException(
            status_code=503,
            detail=f"Knowledge base not ready: {str(e)}. Please call POST /ingest first.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")
