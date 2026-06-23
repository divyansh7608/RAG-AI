# RAG AI Knowledge Chatbot

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot built with:

- 🧠 **Gemini 3 Flash Preview** — LLM for answer generation
- 🔢 **Gemini embedding-001** — Semantic embedding
- 🗄️ **ChromaDB** — Local persistent vector store
- ⚡ **FastAPI** — Python backend API
- 🌐 **Vanilla HTML/CSS/JS** — No-framework frontend

---

## Project Structure

```
RAG_AI/
├── backend/
│   ├── main.py            # FastAPI app (/health, /ingest, /chat)
│   ├── scraper.py         # Scrapes Google AI Blog + OpenAI Blog
│   ├── pdf_generator.py   # Generates ai_knowledge.pdf
│   ├── ingestion.py       # Chunking + embedding + ChromaDB storage
│   ├── retriever.py       # Query embedding + vector search
│   └── chat.py            # RAG chat with Gemini 3 Flash Preview
├── frontend/
│   ├── index.html         # Single-page chat UI
│   ├── app.js             # Frontend logic (no frameworks)
│   └── style.css          # Dark glassmorphism design
├── vector_store/          # Auto-created by ChromaDB on first ingest
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

> **Important — Python environment:**  
> You may have multiple Python installations (e.g. Anaconda + system Python).  
> Always use `python -m pip install` (not bare `pip install`) to ensure packages
> go into the same Python that runs your scripts.

---

### Step 1 — Install dependencies

```powershell
python -m pip install -r requirements.txt
```

> ✅ Use `python -m pip`, NOT just `pip`, to avoid environment mismatch errors.

---

### Step 2 — Add your Gemini API key

```powershell
# Create .env from the example
copy .env.example .env
```

Open `.env` and set your key:

```
GEMINI_API_KEY=your_actual_gemini_key_here
```

> Get a free key at 👉 https://aistudio.google.com

---

### Step 3 — Generate the knowledge base PDF

```powershell
python backend/pdf_generator.py
```

Expected output:
```
PDF generated successfully: D:\Project\RAG_AI\backend\ai_knowledge.pdf
```

---

### Step 4 — Run the ingestion pipeline

```powershell
python backend/ingestion.py
```

This will:
- Parse `ai_knowledge.pdf` with PyMuPDF
- Scrape Google AI Blog & OpenAI Blog (with auto-fallback if blocked)
- Chunk all text (~500 tokens per chunk, 50-token overlap)
- Embed chunks with Gemini `embedding-001`
- Store everything in `./vector_store/` via ChromaDB

> ⏱️ Takes 2–5 minutes on first run (Gemini API rate limits).  
> It is **safe to re-run** — ChromaDB uses upsert so no duplicates.

---

### Step 5 — Start the FastAPI backend

```powershell
python -m uvicorn backend.main:app --reload
```

Server starts at: **http://127.0.0.1:8000**

Interactive API docs: **http://127.0.0.1:8000/docs**

> ✅ Use `python -m uvicorn` (not just `uvicorn`) to avoid PATH issues.

---

### Step 6 — Open the frontend

Open `frontend/index.html` directly in your browser — **no web server needed**.

The status indicator in the sidebar will turn green once the backend is reachable.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health`  | Returns `{"status": "ok"}` |
| `POST` | `/ingest`  | Runs full ingestion pipeline |
| `POST` | `/chat`    | `{"query": "..."}` → `{"answer": "...", "sources": [...]}` |

---

## How It Works

```
User Query
    |
    v
embed_query()           <-- Gemini embedding-001 (retrieval_query)
    |
    v
ChromaDB ANN search     <-- top-5 most similar chunks (cosine similarity)
    |
    v
Build RAG prompt        <-- system instructions + context chunks + question
    |
    v
Gemini 3 Flash Preview    <-- grounded answer generation
    |
    v
Return answer + sources
```

---

## Changing the Backend URL

If your server runs on a different host or port, edit the top of `frontend/app.js`:

```js
const CONFIG = {
  BACKEND_URL: "http://127.0.0.1:8000",  // <-- change this
};
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` after `pip install` | Use `python -m pip install -r requirements.txt` instead of bare `pip` |
| `GEMINI_API_KEY not set` | Create `.env` file from `.env.example` and add your key |
| `Vector store not found` | Run `python backend/ingestion.py` first |
| Frontend shows "Backend offline" | Start server: `python -m uvicorn backend.main:app --reload` |
| `UnicodeEncodeError` in pdf_generator | Fixed — sanitize() strips non-Latin-1 chars automatically |
| Scraping returns 0 articles | Fallback content is used automatically |
| Rate limit errors during embed | Wait 60s, then re-run ingestion (uses upsert, safe to retry) |

---

## Tech Stack

| Component | Library |
|-----------|---------|
| LLM | `google-generativeai` — gemini-3-flash-preview |
| Embeddings | `google-generativeai` — embedding-001 |
| Vector DB | `chromadb` (local, file-based) |
| API Server | `fastapi` + `uvicorn` |
| PDF generate | `fpdf2` |
| PDF parse | `pymupdf` |
| Web scrape | `beautifulsoup4` + `httpx` |
| Env config | `python-dotenv` |
