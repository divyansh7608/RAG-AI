"""
retriever.py
------------
Retrieves the most relevant text chunks from ChromaDB for a given query.

Steps:
  1. Embed the user query with Gemini embedding-001
  2. Query ChromaDB for the top-k most similar chunks
  3. Return a list of {text, source} dicts
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Environment & API setup
# ─────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set. Please add it to your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

# ─────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
VECTOR_STORE_DIR = os.path.join(PROJECT_DIR, "vector_store")
COLLECTION_NAME = "ai_knowledge"
TOP_K = 5  # number of chunks to retrieve


# ─────────────────────────────────────────────
# ChromaDB client (lazy singleton)
# ─────────────────────────────────────────────
_chroma_client = None
_collection = None


def get_collection():
    """Return the ChromaDB collection, initialising the client if needed."""
    global _chroma_client, _collection
    if _collection is None:
        if not os.path.exists(VECTOR_STORE_DIR):
            raise RuntimeError(
                f"Vector store not found at '{VECTOR_STORE_DIR}'. "
                "Please run ingestion.py first."
            )
        _chroma_client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ─────────────────────────────────────────────
# Embed query
# ─────────────────────────────────────────────
def embed_query(query: str) -> list[float]:
    """
    Embed the user's query using Gemini embedding-001.
    Uses task_type='retrieval_query' (asymmetric retrieval).
    """
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=query,
        task_type="retrieval_query",
    )
    return result["embedding"]


# ─────────────────────────────────────────────
# Retrieve top-k chunks
# ─────────────────────────────────────────────
def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Retrieve the top_k most relevant chunks for the given query.

    Returns:
        A list of dicts, each with:
          - text   : the chunk text content
          - source : source document / URL
          - title  : document title
          - score  : cosine similarity distance (lower = more similar)
    """
    try:
        collection = get_collection()

        # Validate collection is not empty
        count = collection.count()
        if count == 0:
            raise RuntimeError("Vector store is empty. Please run ingestion.py first.")

        # Embed the query
        query_embedding = embed_query(query)

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),  # can't ask for more than what's stored
            include=["documents", "metadatas", "distances"],
        )

        # Parse results into a clean list
        retrieved = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            retrieved.append({
                "text": doc,
                "source": meta.get("source", "Unknown"),
                "title": meta.get("title", "Unknown"),
                "score": round(float(dist), 4),
            })

        return retrieved

    except Exception as e:
        print(f"[Retriever] Error during retrieval: {e}")
        raise


# ─────────────────────────────────────────────
# Run standalone
# ─────────────────────────────────────────────
if __name__ == "__main__":
    test_query = "What is Retrieval-Augmented Generation?"
    print(f"\nQuery: {test_query}\n{'='*60}")

    chunks = retrieve(test_query)
    for i, chunk in enumerate(chunks, 1):
        print(f"\n[{i}] Source : {chunk['source']}")
        print(f"     Score  : {chunk['score']}")
        print(f"     Preview: {chunk['text'][:300]}...")
