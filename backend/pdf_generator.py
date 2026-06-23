"""
pdf_generator.py
----------------
Generates a simulated AI knowledge base PDF using fpdf2.
Covers topics: LLMs, RAG, Transformers, Prompt Engineering,
               Vector Databases, AI Agents.

NOTE: Uses fpdf2's built-in Helvetica core font which supports
      Latin-1 (cp1252) only. All text is sanitised to ASCII/Latin-1
      before rendering to avoid UnicodeEncodeError.

Run standalone:
    python backend/pdf_generator.py
"""

from fpdf import FPDF
import os

# ─────────────────────────────────────────────
# Unicode -> ASCII/Latin-1 sanitiser
# fpdf2 core fonts (Helvetica, Times, Courier) only support Latin-1.
# Replace any character outside that range with a safe ASCII equivalent.
# ─────────────────────────────────────────────
_UNICODE_REPLACEMENTS = {
    "\u2192": "->",    # → RIGHT ARROW
    "\u2190": "<-",    # <- LEFT ARROW
    "\u2194": "<->",   # <-> LEFT RIGHT ARROW
    "\u2018": "'",     # ' LEFT SINGLE QUOTATION MARK
    "\u2019": "'",     # ' RIGHT SINGLE QUOTATION MARK
    "\u201c": '"',     # " LEFT DOUBLE QUOTATION MARK
    "\u201d": '"',     # " RIGHT DOUBLE QUOTATION MARK
    "\u2013": "-",     # - EN DASH
    "\u2014": "--",    # -- EM DASH
    "\u2026": "...",   # ... ELLIPSIS
    "\u00b2": "^2",    # ^2 SUPERSCRIPT TWO
    "\u00b3": "^3",    # ^3 SUPERSCRIPT THREE
    "\u00b7": "-",     # - MIDDLE DOT
    "\u2265": ">=",    # >= GREATER-THAN OR EQUAL TO
    "\u2264": "<=",    # <= LESS-THAN OR EQUAL TO
    "\u00d7": "x",     # x MULTIPLICATION SIGN
    "\u00e9": "e",     # e e WITH ACUTE (accent)
    "\u2022": "-",     # - BULLET
    "\u25ba": ">",     # > BLACK RIGHT-POINTING POINTER
}


def sanitize(text: str) -> str:
    """Replace known non-Latin-1 Unicode characters with ASCII equivalents.
    Any remaining unmappable characters are dropped (errors='ignore')."""
    for char, replacement in _UNICODE_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    # Final safety net: drop anything still outside Latin-1
    return text.encode("latin-1", errors="ignore").decode("latin-1")


# ─────────────────────────────────────────────
# Knowledge base content
# All strings use plain ASCII / Latin-1 characters only.
# ─────────────────────────────────────────────
SECTIONS = [
    {
        "title": "Large Language Models (LLMs)",
        "content": (
            "Large Language Models (LLMs) are deep learning models trained on massive text corpora "
            "using the Transformer architecture. They learn statistical patterns in language and can "
            "generate coherent, contextually relevant text. Examples include GPT-4, Gemini, Claude, "
            "and LLaMA. LLMs are pre-trained on internet-scale data and fine-tuned for specific tasks.\n\n"
            "Key properties of LLMs:\n"
            "- Emergent capabilities: abilities that appear only at large scale (e.g., chain-of-thought reasoning).\n"
            "- In-context learning: the model adapts to a task purely from examples in the prompt, without weight updates.\n"
            "- Hallucination: LLMs can confidently produce factually incorrect outputs, making retrieval augmentation important.\n"
            "- Tokenization: text is split into subword tokens; GPT-4 uses ~100k vocabulary BPE tokens.\n\n"
            "Training pipeline: pre-training (next-token prediction on raw text) -> supervised fine-tuning (SFT) -> "
            "reinforcement learning from human feedback (RLHF) or direct preference optimization (DPO). "
            "Inference is auto-regressive: one token is sampled at a time conditioned on all previous tokens."
        ),
    },
    {
        "title": "Retrieval-Augmented Generation (RAG)",
        "content": (
            "Retrieval-Augmented Generation (RAG) combines a neural retriever with a generative LLM. "
            "Instead of relying solely on parametric knowledge baked into model weights, RAG fetches "
            "relevant documents at query time and feeds them into the LLM context window.\n\n"
            "RAG Pipeline Steps:\n"
            "1. Ingestion: Documents are chunked, embedded, and stored in a vector database.\n"
            "2. Retrieval: The user query is embedded; top-k similar chunks are retrieved via ANN search.\n"
            "3. Augmentation: Retrieved chunks are prepended to the LLM prompt as context.\n"
            "4. Generation: The LLM generates an answer grounded in the retrieved context.\n\n"
            "Benefits of RAG:\n"
            "- Reduces hallucination by grounding answers in real documents.\n"
            "- Enables real-time knowledge updates without model retraining.\n"
            "- Provides source attribution (cite which documents were used).\n"
            "- Cost-effective compared to fine-tuning for knowledge injection.\n\n"
            "Advanced RAG techniques include HyDE (Hypothetical Document Embeddings), query rewriting, "
            "re-ranking with cross-encoders, and multi-hop retrieval chains."
        ),
    },
    {
        "title": "Transformer Architecture",
        "content": (
            "The Transformer, introduced in 'Attention Is All You Need' (Vaswani et al., 2017), replaced "
            "RNNs for sequence modelling. Its core mechanism is multi-head self-attention, which allows "
            "every token to attend to every other token in O(n^2) time.\n\n"
            "Key components:\n"
            "- Positional Encoding: Since Transformers lack recurrence, sinusoidal or learned positional "
            "  embeddings inject sequence order information.\n"
            "- Multi-Head Self-Attention: Q, K, V projections are computed in parallel across h heads, "
            "  enabling the model to attend to different representation subspaces simultaneously.\n"
            "- Feed-Forward Network (FFN): Two linear layers with a non-linearity (GELU/ReLU) applied "
            "  position-wise after the attention sublayer.\n"
            "- Layer Normalization and Residual Connections: Stabilize training in deep networks.\n\n"
            "Variants:\n"
            "- Encoder-only (BERT): masked language modelling; used for classification and embeddings.\n"
            "- Decoder-only (GPT): causal language modelling; used for generation.\n"
            "- Encoder-Decoder (T5, BART): sequence-to-sequence tasks like translation and summarization.\n\n"
            "Scaling laws (Hoffmann et al., Chinchilla 2022) show optimal compute allocation requires "
            "scaling data and parameters proportionally."
        ),
    },
    {
        "title": "Prompt Engineering",
        "content": (
            "Prompt Engineering is the practice of crafting inputs to LLMs to elicit desired outputs "
            "without modifying model weights. Effective prompting can dramatically improve accuracy "
            "on reasoning, coding, and knowledge tasks.\n\n"
            "Core techniques:\n"
            "- Zero-shot prompting: Ask the model directly without examples.\n"
            "- Few-shot prompting: Provide 2-8 input-output demonstrations before the query.\n"
            "- Chain-of-Thought (CoT): Instruct the model to 'think step by step', enabling multi-step reasoning.\n"
            "- Self-Consistency: Sample multiple CoT paths and majority-vote the final answer.\n"
            "- Tree-of-Thoughts (ToT): Explore multiple reasoning branches and backtrack.\n"
            "- ReAct: Interleave reasoning (Thought) with actions (tool calls) and observations.\n"
            "- Role prompting: Assign a persona ('You are an expert data scientist...').\n\n"
            "System prompt best practices:\n"
            "- Be explicit about output format (JSON, markdown, bullet points).\n"
            "- Specify constraints (word count, language, tone).\n"
            "- Separate instructions from content using delimiters (triple backticks, XML tags).\n"
            "- Instruct the model to cite sources when using RAG.\n\n"
            "Prompt injection attacks (malicious instructions embedded in retrieved documents) are an "
            "important security concern in RAG systems."
        ),
    },
    {
        "title": "Vector Databases",
        "content": (
            "Vector databases store high-dimensional embedding vectors and enable fast approximate "
            "nearest-neighbour (ANN) search. They are the storage backbone of RAG systems.\n\n"
            "How embeddings work:\n"
            "Text is passed through an embedding model (e.g., Gemini embedding-001, OpenAI text-embedding-3) "
            "to produce a dense floating-point vector (e.g., 768 or 1536 dimensions). Semantically similar "
            "texts cluster close together in this embedding space as measured by cosine similarity or dot product.\n\n"
            "Popular vector databases:\n"
            "- ChromaDB: Open-source, embedded Python library, great for prototyping. Supports persistent "
            "  disk storage and in-memory modes.\n"
            "- Pinecone: Managed cloud service, scalable to billions of vectors.\n"
            "- Weaviate: Open-source with hybrid BM25 + vector search.\n"
            "- Qdrant: High-performance Rust-based vector DB with payload filtering.\n"
            "- FAISS: Facebook's library for CPU/GPU ANN search; not a full database.\n\n"
            "ANN algorithms used:\n"
            "- HNSW (Hierarchical Navigable Small World): graph-based, high recall, used by ChromaDB/Weaviate.\n"
            "- IVF (Inverted File Index): clustering-based, used by FAISS.\n"
            "- Product Quantization (PQ): compresses vectors for memory efficiency.\n\n"
            "Metadata filtering allows pre/post-filtering of results by attributes (date, source, category) "
            "before or after the ANN search."
        ),
    },
    {
        "title": "AI Agents",
        "content": (
            "AI Agents are LLM-powered systems that autonomously plan and execute multi-step tasks by "
            "calling tools, browsing the web, writing code, and coordinating with other agents.\n\n"
            "Agent architectures:\n"
            "- ReAct (Reason + Act): The model alternates between reasoning steps and tool invocations, "
            "  observing results until the task is complete.\n"
            "- Plan-and-Execute: A planner LLM generates a high-level task list; executor agents handle "
            "  individual subtasks.\n"
            "- Multi-Agent Systems: Orchestrator delegates subtasks to specialised sub-agents "
            "  (researcher, coder, critic).\n\n"
            "Common tools agents use:\n"
            "- Web search (Google Search API, Bing)\n"
            "- Code interpreter (Python sandbox execution)\n"
            "- File system read/write\n"
            "- RAG retrieval over a knowledge base\n"
            "- API calls (calendar, email, databases)\n\n"
            "Frameworks: LangChain, LangGraph, AutoGen, CrewAI, and Google's Agent Development Kit (ADK).\n\n"
            "Key challenges:\n"
            "- Long-horizon planning: maintaining coherent goals across many steps.\n"
            "- Tool reliability: gracefully handling tool errors and unexpected outputs.\n"
            "- Memory: episodic (conversation history), semantic (knowledge base), and procedural memory.\n"
            "- Safety: avoiding harmful actions in open-ended agentic loops."
        ),
    },
]

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "ai_knowledge.pdf")


# ─────────────────────────────────────────────
# PDF class with header and footer
# ─────────────────────────────────────────────
class KnowledgePDF(FPDF):
    """Custom PDF with branded header and page footer."""

    def header(self):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(60, 90, 180)
        self.cell(0, 10, "AI Knowledge Base", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(60, 90, 180)
        self.set_line_width(0.5)
        self.line(10, 22, 200, 22)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, sanitize(f"Page {self.page_no()}"), align="C")


# ─────────────────────────────────────────────
# PDF generation
# ─────────────────────────────────────────────
def generate_pdf(output_path: str = OUTPUT_PATH) -> str:
    """Generate the AI knowledge base PDF and save it to output_path."""
    pdf = KnowledgePDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)

    # ── Cover page ──────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(30, 30, 60)
    pdf.ln(40)
    pdf.cell(0, 20, "AI Knowledge Base", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(80, 80, 120)
    pdf.cell(
        0, 10,
        "A Comprehensive Reference for RAG-Powered Chatbots",
        align="C", new_x="LMARGIN", new_y="NEXT",
    )
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(120, 120, 150)
    # Use plain ASCII separator (| instead of middle-dot ·)
    pdf.cell(
        0, 8,
        "Topics: LLMs | RAG | Transformers | Prompt Engineering | Vector Databases | AI Agents",
        align="C", new_x="LMARGIN", new_y="NEXT",
    )

    # ── Content pages ────────────────────────────────────────────────────────
    for section in SECTIONS:
        pdf.add_page()

        # Section title — sanitise before rendering
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(40, 70, 160)
        pdf.cell(0, 12, sanitize(section["title"]), new_x="LMARGIN", new_y="NEXT")

        # Divider line
        pdf.set_draw_color(40, 70, 160)
        pdf.set_line_width(0.4)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(4)

        # Body text — sanitise before rendering
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(30, 30, 30)
        pdf.multi_cell(0, 7, sanitize(section["content"]))

    pdf.output(output_path)
    return output_path


# ─────────────────────────────────────────────
# Run standalone
# ─────────────────────────────────────────────
if __name__ == "__main__":
    path = generate_pdf()
    print(f"PDF generated successfully: {path}")
