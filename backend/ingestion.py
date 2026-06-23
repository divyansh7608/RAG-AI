"""
ingestion.py
------------
Ingestion pipeline for the RAG chatbot:
  1. Load text from ai_knowledge.pdf (PyMuPDF)
  2. Load scraped blog articles (scraper.py)
  3. Chunk text into ~500-token windows with 50-token overlap
  4. Embed each chunk with Gemini embedding-001
  5. Store embeddings in ChromaDB (persisted to ./vector_store)

Run standalone:
    python ingestion.py
"""

import os
import sys
import time

# Allow imports from the backend package when run standalone
sys.path.insert(0, os.path.dirname(__file__))

import fitz  # PyMuPDF
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from scraper import scrape_all

# ─────────────────────────────────────────────
# Environment & API setup
# ─────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set. Please add it to your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
PDF_PATH = os.path.join(BACKEND_DIR, "ai_knowledge.pdf")
VECTOR_STORE_DIR = os.path.join(PROJECT_DIR, "vector_store")
COLLECTION_NAME = "ai_knowledge"

# ─────────────────────────────────────────────
# Chunking configuration
# ─────────────────────────────────────────────
# Approximate tokens ≈ words * 1.3; we use word count as proxy.
CHUNK_SIZE_WORDS = 380   # ~500 tokens
OVERLAP_WORDS = 38       # ~50 tokens


# ─────────────────────────────────────────────
# Step 1: Load PDF text with PyMuPDF
# ─────────────────────────────────────────────
def load_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from each page of the PDF.
    Returns a list of {source, title, content} dicts (one per page).
    """
    if not os.path.exists(pdf_path):
        print(f"⚠ PDF not found at {pdf_path}. Skipping PDF ingestion.")
        print("  Run: python backend/pdf_generator.py")
        return []

    print(f"\n[PDF] Loading: {pdf_path}")
    doc = fitz.open(pdf_path)
    pages_data = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if text:
            pages_data.append({
                "source": f"ai_knowledge.pdf (page {page_num + 1})",
                "title": f"AI Knowledge Base - Page {page_num + 1}",
                "content": text,
            })

    doc.close()
    print(f"[PDF] Extracted {len(pages_data)} pages of text.")
    return pages_data


# ─────────────────────────────────────────────
# Step 2: Chunk text with overlap
# ─────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    Each chunk has ~chunk_size words with an overlap of ~overlap words.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += chunk_size - overlap  # slide forward with overlap

    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk all documents and return a flat list of chunk dicts:
    {chunk_id, source, title, content}
    """
    all_chunks = []
    for doc_idx, doc in enumerate(documents):
        chunks = chunk_text(doc["content"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"doc_{doc_idx}__chunk_{i}",
                "source": doc["source"],
                "title": doc["title"],
                "content": chunk,
            })
    return all_chunks


# ─────────────────────────────────────────────
# Step 3: Embed chunks with Gemini
# ─────────────────────────────────────────────
def embed_text(text: str, task_type: str = "retrieval_document") -> list[float]:
    """
    Embed a single text string using Gemini embedding-001.
    task_type: 'retrieval_document' for documents, 'retrieval_query' for queries.
    """
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        task_type=task_type,
    )
    return result["embedding"]


def embed_chunks_batch(chunks: list[dict], batch_size: int = 20) -> list[list[float]]:
    """
    Embed all chunks in batches to respect API rate limits.
    Returns a list of embedding vectors corresponding to each chunk.
    """
    embeddings = []
    total = len(chunks)

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        print(f"  Embedding chunks {i+1}-{min(i+batch_size, total)} / {total}...")

        for chunk in batch:
            try:
                emb = embed_text(chunk["content"])
                embeddings.append(emb)
            except Exception as e:
                print(f"  X Embedding failed for chunk '{chunk['chunk_id']}': {e}")
                # Use a zero vector as placeholder so indices stay aligned
                embeddings.append([0.0] * 768)

        # Respect Gemini API rate limits (60 req/min on free tier)
        if i + batch_size < total:
            time.sleep(1)

    return embeddings


# ─────────────────────────────────────────────
# Step 4: Store in ChromaDB
# ─────────────────────────────────────────────
def get_chroma_collection():
    """
    Return (or create) a persistent ChromaDB collection.
    Embeddings are stored to disk at VECTOR_STORE_DIR.
    """
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
    # get_or_create_collection: safe to call even if collection already exists
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # use cosine similarity
    )
    return collection


def store_chunks(collection, chunks: list[dict], embeddings: list[list[float]]):
    """
    Upsert chunks and their embeddings into ChromaDB.
    Uses upsert so re-running ingestion is idempotent.
    """
    ids = [c["chunk_id"] for c in chunks]
    documents = [c["content"] for c in chunks]
    metadatas = [{"source": c["source"], "title": c["title"]} for c in chunks]

    # ChromaDB upsert in batches of 100 (Chroma default max is 5461 per call)
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )
        print(f"  Stored chunks {i+1}-{min(i+batch_size, len(ids))} / {len(ids)} in ChromaDB.")


# ─────────────────────────────────────────────
# Main ingestion pipeline
# ─────────────────────────────────────────────
def run_ingestion() -> dict:
    """
    Full ingestion pipeline: load → chunk → embed → store.
    Returns a status dict with counts.
    """
    print("\n" + "=" * 60)
    print("  RAG Ingestion Pipeline")
    print("=" * 60)

    # 1. Load data sources
    pdf_docs = load_pdf(PDF_PATH)

    print("\n[Scraper] Fetching blog articles...")
    blog_docs = scrape_all()

    all_docs = pdf_docs + blog_docs
    print(f"\n[Ingestion] Total documents loaded: {len(all_docs)}")

    if not all_docs:
        return {"status": "error", "message": "No documents found to ingest."}

    # 2. Chunk documents
    print("\n[Chunking] Splitting documents into chunks...")
    chunks = chunk_documents(all_docs)
    print(f"[Chunking] Total chunks: {len(chunks)}")

    # 3. Embed chunks
    print("\n[Embedding] Generating embeddings with Gemini gemini-embedding-001...")
    embeddings = embed_chunks_batch(chunks)
    print(f"[Embedding] Generated {len(embeddings)} embeddings.")

    # 4. Store in ChromaDB
    print("\n[ChromaDB] Storing chunks in vector store...")
    collection = get_chroma_collection()
    store_chunks(collection, chunks, embeddings)

    final_count = collection.count()
    print(f"\nOK Ingestion complete! Total chunks in vector store: {final_count}")
    print(f"   Vector store location: {VECTOR_STORE_DIR}")

    return {
        "status": "success",
        "documents_loaded": len(all_docs),
        "chunks_created": len(chunks),
        "total_in_store": final_count,
    }


# ─────────────────────────────────────────────
# Run standalone
# ─────────────────────────────────────────────
if __name__ == "__main__":
    result = run_ingestion()
    print(f"\nResult: {result}")
