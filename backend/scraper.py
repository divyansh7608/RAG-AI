"""
scraper.py
----------
Scrapes the latest AI blog posts from:
  1. https://blog.google/technology/ai/  (Google AI Blog)
  2. https://openai.com/news/            (OpenAI Blog)

Returns a list of dicts: {source, title, content}

Handles errors gracefully (timeouts, blocks, parsing failures).
"""

import httpx
from bs4 import BeautifulSoup
from typing import List, Dict
import time

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

TIMEOUT = 15  # seconds per request
MAX_ARTICLES = 5  # articles to scrape per source


# ─────────────────────────────────────────────
# Helper: Clean raw text
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Remove excess whitespace and blank lines from scraped text."""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]  # drop empty lines
    return " ".join(lines)


# ─────────────────────────────────────────────
# Scraper: Google AI Blog
# ─────────────────────────────────────────────
def scrape_google_ai_blog(client: httpx.Client) -> List[Dict]:
    """
    Scrape the Google AI Blog listing page and fetch article content.
    URL: https://blog.google/technology/ai/
    """
    results = []
    base_url = "https://blog.google"
    listing_url = f"{base_url}/technology/ai/"

    try:
        print(f"  -> Fetching Google AI Blog listing: {listing_url}")
        resp = client.get(listing_url, timeout=TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Google blog uses <article> tags or <h3>/<a> links in the listing
        # Try multiple selector patterns for resilience
        article_links = []

        # Pattern 1: article cards with a link
        for article in soup.select("article a[href]"):
            href = article.get("href", "")
            if "/technology/ai/" in href and href not in article_links:
                full_url = base_url + href if href.startswith("/") else href
                article_links.append(full_url)

        # Pattern 2: h3 links inside post-feed sections
        if not article_links:
            for tag in soup.select("h3 a[href], h2 a[href]"):
                href = tag.get("href", "")
                if "/technology/ai/" in href and href not in article_links:
                    full_url = base_url + href if href.startswith("/") else href
                    article_links.append(full_url)

        article_links = list(dict.fromkeys(article_links))[:MAX_ARTICLES]

        if not article_links:
            print("  ! No article links found on Google AI Blog listing page.")
            return results

        # Fetch each article page
        for url in article_links:
            try:
                time.sleep(0.5)  # polite delay
                print(f"    -> Fetching article: {url}")
                a_resp = client.get(url, timeout=TIMEOUT)
                a_resp.raise_for_status()
                a_soup = BeautifulSoup(a_resp.text, "html.parser")

                title_tag = a_soup.find("h1")
                title = clean_text(title_tag.get_text()) if title_tag else "Untitled"

                # Extract main article body text
                body_candidates = a_soup.select("article, .article-body, .post-body, main")
                body_text = ""
                if body_candidates:
                    for tag in body_candidates[0].find_all(["p", "li", "h2", "h3"]):
                        body_text += tag.get_text(separator=" ") + " "
                else:
                    body_text = a_soup.get_text(separator=" ")

                content = clean_text(body_text)[:3000]  # cap at 3000 chars per article

                if title and content:
                    results.append({"source": "Google AI Blog", "title": title, "content": content})
                    print(f"    OK Scraped: {title[:60]}")

            except Exception as e:
                print(f"    X Failed to fetch {url}: {e}")

    except Exception as e:
        print(f"  X Google AI Blog listing failed: {e}")

    return results


# ─────────────────────────────────────────────
# Scraper: OpenAI Blog
# ─────────────────────────────────────────────
def scrape_openai_blog(client: httpx.Client) -> List[Dict]:
    """
    Scrape the OpenAI News page and fetch article content.
    URL: https://openai.com/news/
    """
    results = []
    base_url = "https://openai.com"
    listing_url = f"{base_url}/news/"

    try:
        print(f"  -> Fetching OpenAI News listing: {listing_url}")
        resp = client.get(listing_url, timeout=TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # OpenAI uses <a> tags leading to /research/ or /blog/ or /index/
        article_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # Filter for news/research/blog article paths
            if any(seg in href for seg in ["/research/", "/blog/", "/index/"]):
                full_url = base_url + href if href.startswith("/") else href
                if full_url not in article_links:
                    article_links.append(full_url)

        article_links = list(dict.fromkeys(article_links))[:MAX_ARTICLES]

        if not article_links:
            print("  ! No article links found on OpenAI News listing page.")
            return results

        for url in article_links:
            try:
                time.sleep(0.5)
                print(f"    -> Fetching article: {url}")
                a_resp = client.get(url, timeout=TIMEOUT)
                a_resp.raise_for_status()
                a_soup = BeautifulSoup(a_resp.text, "html.parser")

                title_tag = a_soup.find("h1")
                title = clean_text(title_tag.get_text()) if title_tag else "Untitled"

                body_candidates = a_soup.select("article, .content, main, [class*='post']")
                body_text = ""
                if body_candidates:
                    for tag in body_candidates[0].find_all(["p", "li", "h2", "h3"]):
                        body_text += tag.get_text(separator=" ") + " "
                else:
                    body_text = a_soup.get_text(separator=" ")

                content = clean_text(body_text)[:3000]

                if title and content:
                    results.append({"source": "OpenAI Blog", "title": title, "content": content})
                    print(f"    OK Scraped: {title[:60]}")

            except Exception as e:
                print(f"    X Failed to fetch {url}: {e}")

    except Exception as e:
        print(f"  X OpenAI Blog listing failed: {e}")

    return results


# ─────────────────────────────────────────────
# Fallback static content (used when scraping is blocked)
# ─────────────────────────────────────────────
FALLBACK_ARTICLES = [
    {
        "source": "Google AI Blog (fallback)",
        "title": "Gemini 1.5: Our next-generation model",
        "content": (
            "Gemini 1.5 represents a significant step forward in AI capabilities. "
            "The model features a long context window of up to 1 million tokens, enabling "
            "reasoning over entire codebases, books, and lengthy video transcripts. "
            "Gemini 1.5 Pro achieves near-perfect recall on the Needle-in-a-Haystack benchmark "
            "and shows strong performance across multimodal tasks including text, audio, image, and video. "
            "The model uses a Mixture-of-Experts (MoE) architecture to improve efficiency. "
            "Google continues to improve safety measures including red-teaming and constitutional AI techniques."
        ),
    },
    {
        "source": "Google AI Blog (fallback)",
        "title": "Advances in Multimodal AI Research",
        "content": (
            "Multimodal AI systems can process and reason across text, images, audio, and video simultaneously. "
            "Recent advances include models that can follow complex instructions across modalities, "
            "generate images from detailed text descriptions, and understand spoken language in context. "
            "Google DeepMind's research focuses on unified architectures that handle all modalities "
            "within a single model rather than separate specialist systems. "
            "Key benchmarks include VQA (Visual Question Answering), MMMU, and video understanding tasks."
        ),
    },
    {
        "source": "OpenAI Blog (fallback)",
        "title": "GPT-4o: Our most capable and efficient model",
        "content": (
            "GPT-4o ('o' for omni) is OpenAI's flagship multimodal model capable of processing "
            "text, audio, and images natively in a single model. Unlike previous versions that used "
            "separate pipelines for each modality, GPT-4o handles everything end-to-end. "
            "It responds to audio inputs in as little as 232ms, matching human response time in conversation. "
            "The model also shows strong performance on coding, math, and multilingual benchmarks. "
            "GPT-4o is available through the ChatGPT interface and OpenAI API."
        ),
    },
    {
        "source": "OpenAI Blog (fallback)",
        "title": "Introducing ChatGPT and Whisper APIs",
        "content": (
            "OpenAI has made the ChatGPT model available via API, allowing developers to integrate "
            "conversational AI into their own applications. The gpt-3.5-turbo model is optimised for "
            "dialogue and priced 10x cheaper than existing GPT-3.5 models. "
            "The Whisper speech-to-text model is also available as an API, enabling accurate transcription "
            "in 57 languages. Together these APIs unlock a new generation of AI-powered products "
            "spanning customer support, education, productivity, and creative tools."
        ),
    },
    {
        "source": "OpenAI Blog (fallback)",
        "title": "Scaling Laws for Neural Language Models",
        "content": (
            "Research on scaling laws reveals predictable relationships between model performance, "
            "compute budget, dataset size, and number of parameters. The Chinchilla scaling laws suggest "
            "that for a given compute budget, models should be trained on roughly 20 tokens per parameter. "
            "This insight led to more compute-optimal models. Emergent abilities—capabilities that appear "
            "suddenly at certain scales—include multi-step reasoning, arithmetic, and in-context learning. "
            "Understanding these laws helps researchers allocate compute budgets efficiently during training."
        ),
    },
]


# ─────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────
def scrape_all() -> List[Dict]:
    """
    Scrape Google AI Blog and OpenAI Blog.
    Falls back to static content if scraping fails or returns too few articles.
    Returns a list of {source, title, content} dicts.
    """
    articles = []

    with httpx.Client(headers=HEADERS, follow_redirects=True) as client:
        print("\n[Scraper] Scraping Google AI Blog...")
        google_articles = scrape_google_ai_blog(client)
        articles.extend(google_articles)

        print("\n[Scraper] Scraping OpenAI Blog...")
        openai_articles = scrape_openai_blog(client)
        articles.extend(openai_articles)

    # If we got fewer than 3 articles total, supplement with fallback data
    if len(articles) < 3:
        print(
            f"\n[Scraper] Only {len(articles)} article(s) scraped. "
            "Using fallback static articles to supplement."
        )
        existing_titles = {a["title"] for a in articles}
        for fallback in FALLBACK_ARTICLES:
            if fallback["title"] not in existing_titles:
                articles.append(fallback)

    print(f"\n[Scraper] Total articles collected: {len(articles)}")
    return articles


# ─────────────────────────────────────────────
# Run standalone
# ─────────────────────────────────────────────
if __name__ == "__main__":
    data = scrape_all()
    for i, article in enumerate(data, 1):
        print(f"\n{'='*60}")
        print(f"[{i}] Source : {article['source']}")
        print(f"     Title  : {article['title']}")
        print(f"     Length : {len(article['content'])} chars")
        print(f"     Preview: {article['content'][:200]}...")
