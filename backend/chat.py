"""
chat.py
-------
Chat module for the RAG chatbot.

Steps:
  1. Accept a user query
  2. Retrieve relevant chunks via retriever.py
  3. Build a grounded prompt with system context + retrieved chunks + user question
  4. Call Gemini gemini-1.5-flash for the answer
  5. Return the answer string and source list
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import google.generativeai as genai
from dotenv import load_dotenv
from retriever import retrieve

# ─────────────────────────────────────────────
# Environment & API setup
# ─────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set. Please add it to your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

# ─────────────────────────────────────────────
# Gemini model configuration
# ─────────────────────────────────────────────
MODEL_NAME = "gemini-3-flash-preview"

GENERATION_CONFIG = {
    "temperature": 0.3,       # lower = more factual / less creative
    "top_p": 0.9,
    "max_output_tokens": 1024,
}

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# ─────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────
SYSTEM_INSTRUCTION = (
    "You are an expert AI assistant specialising in Artificial Intelligence topics including "
    "Large Language Models, Retrieval-Augmented Generation (RAG), Transformer architecture, "
    "Prompt Engineering, Vector Databases, and AI Agents. "
    "Answer questions using ONLY the context provided below. "
    "If the answer is not found in the context, say 'I don't have enough information in my knowledge base to answer that.' "
    "Be concise, accurate, and cite which context snippet(s) support your answer."
)


def build_prompt(query: str, context_chunks: list[dict]) -> str:
    """
    Assemble the full prompt from system instructions, retrieved context, and the user query.
    """
    context_text = ""
    for i, chunk in enumerate(context_chunks, 1):
        context_text += (
            f"\n--- Context [{i}] ---\n"
            f"Source: {chunk['source']}\n"
            f"Content: {chunk['text']}\n"
        )

    prompt = (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"CONTEXT:\n{context_text}\n"
        f"USER QUESTION: {query}\n\n"
        "ANSWER:"
    )
    return prompt


# ─────────────────────────────────────────────
# Main chat function
# ─────────────────────────────────────────────
def chat(query: str) -> dict:
    """
    Answer a user query using RAG.

    Args:
        query: The user's natural language question.

    Returns:
        A dict with:
          - answer  : the generated answer string
          - sources : list of source strings used for grounding
    """
    if not query or not query.strip():
        return {"answer": "Please ask a question.", "sources": []}

    try:
        # Step 1: Retrieve relevant chunks
        chunks = retrieve(query)

        if not chunks:
            return {
                "answer": "I couldn't find relevant information in the knowledge base. "
                          "Please try rephrasing your question.",
                "sources": [],
            }

        # Step 2: Build grounded prompt
        prompt = build_prompt(query, chunks)

        # Step 3: Call Gemini gemini-1.5-flash
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=GENERATION_CONFIG,
            safety_settings=SAFETY_SETTINGS,
        )
        response = model.generate_content(prompt)

        # Step 4: Extract answer text
        answer = response.text.strip() if response.text else "No response generated."

        # Step 5: Deduplicate sources
        sources = list(dict.fromkeys(chunk["source"] for chunk in chunks))

        return {"answer": answer, "sources": sources}

    except Exception as e:
        error_msg = f"An error occurred while generating the answer: {str(e)}"
        print(f"[Chat] Error: {error_msg}")
        return {"answer": error_msg, "sources": []}


# ─────────────────────────────────────────────
# Run standalone
# ─────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "What is Retrieval-Augmented Generation and how does it work?",
        "Explain the Transformer architecture.",
        "What are AI Agents and what tools do they use?",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = chat(q)
        print(f"\nA: {result['answer']}")
        print(f"\nSources: {result['sources']}")
