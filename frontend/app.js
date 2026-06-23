/**
 * app.js  —  RAG AI Knowledge Chatbot frontend
 * Vanilla JS, no frameworks, no npm required.
 *
 * Communicates with the FastAPI backend (main.py).
 * Change BACKEND_URL below if your server runs on a different host/port.
 */

// ─────────────────────────────────────────────
// ⚙️  Config — change this to match your server
// ─────────────────────────────────────────────
const CONFIG = {
  BACKEND_URL: "http://127.0.0.1:8000",  // FastAPI dev server
  HEALTH_POLL_INTERVAL_MS: 15000,         // re-check health every 15s
};

// ─────────────────────────────────────────────
// DOM refs
// ─────────────────────────────────────────────
const messagesContainer = document.getElementById("messagesContainer");
const queryInput        = document.getElementById("queryInput");
const sendBtn           = document.getElementById("sendBtn");
const ingestBtn         = document.getElementById("ingestBtn");
const clearBtn          = document.getElementById("clearBtn");
const statusDot         = document.getElementById("statusDot");
const statusLabel       = document.getElementById("statusLabel");
const welcomeCard       = document.getElementById("welcomeCard");
const ingestOverlay     = document.getElementById("ingestOverlay");
const ingestStatus      = document.getElementById("ingestStatus");
const ingestFill        = document.getElementById("ingestFill");
const menuToggle        = document.getElementById("menuToggle");
const sidebar           = document.querySelector(".sidebar");
const themeToggle       = document.getElementById("themeToggle");
const sunIcon           = document.querySelector(".sun-icon");
const moonIcon          = document.querySelector(".moon-icon");

// ─────────────────────────────────────────────
// State
// ─────────────────────────────────────────────
let isBackendReady = false;   // true once /health returns ok
let isSending      = false;   // prevent double-sends

// ─────────────────────────────────────────────
// ── Health check ─────────────────────────────
// ─────────────────────────────────────────────
async function checkHealth() {
  setStatus("checking", "Checking backend…");
  try {
    const res = await fetch(`${CONFIG.BACKEND_URL}/health`, {
      signal: AbortSignal.timeout(5000),
    });
    if (res.ok) {
      setStatus("ok", "Backend ready");
      isBackendReady = true;
    } else {
      setStatus("error", "Backend error");
    }
  } catch {
    setStatus("error", "Backend offline");
    isBackendReady = false;
  }
}

function setStatus(state, label) {
  statusDot.className = "status-dot " + state;
  statusLabel.textContent = label;
}

// Poll periodically
checkHealth();
setInterval(checkHealth, CONFIG.HEALTH_POLL_INTERVAL_MS);

// ─────────────────────────────────────────────
// ── Input handling ────────────────────────────
// ─────────────────────────────────────────────
queryInput.addEventListener("input", () => {
  // Auto-grow textarea
  queryInput.style.height = "auto";
  queryInput.style.height = Math.min(queryInput.scrollHeight, 140) + "px";

  // Enable/disable send button
  const hasText = queryInput.value.trim().length > 0;
  sendBtn.classList.toggle("ready", hasText && !isSending);
  sendBtn.disabled = !hasText || isSending;
});

queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!isSending && queryInput.value.trim()) handleSend();
  }
});

sendBtn.addEventListener("click", () => {
  if (!isSending && queryInput.value.trim()) handleSend();
});

// ─────────────────────────────────────────────
// ── Send message ──────────────────────────────
// ─────────────────────────────────────────────
async function handleSend() {
  const query = queryInput.value.trim();
  if (!query || isSending) return;

  // Lock UI
  isSending = true;
  sendBtn.classList.remove("ready");
  sendBtn.disabled = true;

  // Clear input
  queryInput.value = "";
  queryInput.style.height = "auto";

  // Hide welcome card
  if (welcomeCard) welcomeCard.remove();

  // Append user message
  appendMessage("user", query);

  // Show typing indicator
  const typingEl = appendTypingIndicator();

  // Call backend
  try {
    if (!isBackendReady) {
      await checkHealth();
      if (!isBackendReady) throw new Error("Backend is offline. Start the server with: uvicorn backend.main:app --reload");
    }

    const res = await fetch(`${CONFIG.BACKEND_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
      signal: AbortSignal.timeout(30000),
    });

    const data = await res.json();

    typingEl.remove();

    if (!res.ok) {
      throw new Error(data.detail || `Server error ${res.status}`);
    }

    appendMessage("bot", data.answer, data.sources || []);

  } catch (err) {
    typingEl.remove();
    const msg = err.message.includes("timed out")
      ? "Request timed out. The model may be taking too long — please try again."
      : err.message;
    appendMessage("bot", msg, [], true);
  }

  // Unlock UI
  isSending = false;
  sendBtn.classList.add("ready");
  sendBtn.disabled = false;
  queryInput.focus();
}

// ─────────────────────────────────────────────
// ── Message builders ──────────────────────────
// ─────────────────────────────────────────────
function appendMessage(role, text, sources = [], isError = false) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "U" : "AI";

  const bubble = document.createElement("div");
  bubble.className = "bubble" + (isError ? " error-bubble" : "");

  // Render newlines + basic markdown-lite (bold **text**)
  bubble.innerHTML = formatText(text);

  // Sources
  if (sources && sources.length > 0) {
    const sourcesEl = document.createElement("div");
    sourcesEl.className = "sources-list";
    const heading = document.createElement("span");
    heading.className = "sources-heading";
    heading.textContent = "Sources";
    sourcesEl.appendChild(heading);
    sources.forEach((src) => {
      const tag = document.createElement("span");
      tag.className = "source-tag";
      tag.textContent = truncateSource(src);
      tag.title = src;
      sourcesEl.appendChild(tag);
    });
    bubble.appendChild(sourcesEl);
  }

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  messagesContainer.appendChild(wrapper);
  scrollToBottom();

  return wrapper;
}

function appendTypingIndicator() {
  const wrapper = document.createElement("div");
  wrapper.className = "message bot";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = "AI";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = `
    <div class="typing-indicator">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>`;

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  messagesContainer.appendChild(wrapper);
  scrollToBottom();
  return wrapper;
}

// ─────────────────────────────────────────────
// ── Formatting helpers ────────────────────────
// ─────────────────────────────────────────────
function formatText(text) {
  // Escape HTML first
  let safe = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // **bold**
  safe = safe.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  // `code`
  safe = safe.replace(/`([^`]+)`/g, `<code style="font-family:var(--font-mono);background:rgba(255,255,255,.06);padding:1px 5px;border-radius:4px;font-size:.85em">$1</code>`);
  // Newlines → <br>
  safe = safe.replace(/\n/g, "<br>");

  return safe;
}

function truncateSource(src, maxLen = 42) {
  // Strip common path prefixes for readability
  src = src.replace(/ai_knowledge\.pdf \(page \d+\)/, (m) => m);
  return src.length > maxLen ? src.slice(0, maxLen) + "…" : src;
}

function scrollToBottom() {
  messagesContainer.scrollTo({ top: messagesContainer.scrollHeight, behavior: "smooth" });
}

// ─────────────────────────────────────────────
// ── Ingest ────────────────────────────────────
// ─────────────────────────────────────────────
ingestBtn.addEventListener("click", async () => {
  if (!isBackendReady) {
    alert("Backend is not reachable. Please start the server first.");
    return;
  }

  // Show overlay
  ingestOverlay.removeAttribute("hidden");
  ingestStatus.textContent = "Loading PDF and scraping blog articles…";
  ingestFill.style.width = "10%";

  // Fake progress animation while waiting
  const progressSteps = [
    [15, "Reading ai_knowledge.pdf…"],
    [30, "Scraping Google AI Blog…"],
    [50, "Scraping OpenAI Blog…"],
    [65, "Chunking documents…"],
    [80, "Generating embeddings with Gemini…"],
    [92, "Storing in ChromaDB…"],
  ];
  let stepIdx = 0;
  const progressTimer = setInterval(() => {
    if (stepIdx < progressSteps.length) {
      const [pct, msg] = progressSteps[stepIdx++];
      ingestFill.style.width = pct + "%";
      ingestStatus.textContent = msg;
    }
  }, 3000);

  try {
    const res = await fetch(`${CONFIG.BACKEND_URL}/ingest`, {
      method: "POST",
      signal: AbortSignal.timeout(300000), // 5 min timeout
    });
    const data = await res.json();

    clearInterval(progressTimer);
    ingestFill.style.width = "100%";
    ingestStatus.textContent = "Complete!";

    await sleep(800);
    ingestOverlay.setAttribute("hidden", "");
    ingestFill.style.width = "0%";

    if (res.ok) {
      appendSystemMessage(`✅ ${data.message}`);
    } else {
      appendSystemMessage(`❌ Ingestion failed: ${data.detail || "Unknown error"}`);
    }
  } catch (err) {
    clearInterval(progressTimer);
    ingestOverlay.setAttribute("hidden", "");
    ingestFill.style.width = "0%";
    appendSystemMessage(`❌ Ingestion error: ${err.message}`);
  }
});

function appendSystemMessage(text) {
  const el = document.createElement("div");
  el.style.cssText = `
    text-align:center; font-size:.78rem; color:var(--clr-text-3);
    padding:8px 16px; border-radius:999px;
    background:var(--clr-surface-3); border:1px solid var(--clr-border);
    margin: 0 auto; max-width: fit-content;
    animation: fadeUp .3s ease both;
  `;
  el.textContent = text;
  messagesContainer.appendChild(el);
  scrollToBottom();
}

// ─────────────────────────────────────────────
// ── Clear chat ────────────────────────────────
// ─────────────────────────────────────────────
clearBtn.addEventListener("click", () => {
  // Remove all message elements (keep welcome card gone)
  messagesContainer.innerHTML = "";
  // Show a fresh welcome card
  const card = document.createElement("div");
  card.className = "welcome-card";
  card.innerHTML = `
    <div class="welcome-icon">
      <svg viewBox="0 0 64 64" fill="none">
        <circle cx="32" cy="32" r="30" stroke="url(#wg2)" stroke-width="2"/>
        <path d="M22 32h20M32 22v20" stroke="url(#wg2)" stroke-width="3" stroke-linecap="round"/>
        <circle cx="32" cy="32" r="6" fill="url(#wg2)" opacity=".2"/>
        <defs>
          <linearGradient id="wg2" x1="2" y1="2" x2="62" y2="62">
            <stop stop-color="#818cf8"/><stop offset="1" stop-color="#34d399"/>
          </linearGradient>
        </defs>
      </svg>
    </div>
    <h2>Ask me anything about AI</h2>
    <p>Chat history cleared. Choose a topic from the sidebar or type your question below.</p>
  `;
  messagesContainer.appendChild(card);
});

// ─────────────────────────────────────────────
// ── Topic chips ───────────────────────────────
// ─────────────────────────────────────────────
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    const q = chip.dataset.query;
    if (q) {
      queryInput.value = q;
      queryInput.dispatchEvent(new Event("input"));
      handleSend();
      // Close sidebar on mobile
      sidebar.classList.remove("open");
    }
  });
});

// ─────────────────────────────────────────────
// ── Theme toggle ──────────────────────────────
// ─────────────────────────────────────────────
const savedTheme = localStorage.getItem("theme") || "dark";
if (savedTheme === "light") {
  document.documentElement.setAttribute("data-theme", "light");
  sunIcon.style.display = "none";
  moonIcon.style.display = "block";
}

themeToggle.addEventListener("click", () => {
  const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
  const newTheme = currentTheme === "dark" ? "light" : "dark";
  
  if (newTheme === "light") {
    document.documentElement.setAttribute("data-theme", "light");
    sunIcon.style.display = "none";
    moonIcon.style.display = "block";
  } else {
    document.documentElement.removeAttribute("data-theme");
    sunIcon.style.display = "block";
    moonIcon.style.display = "none";
  }
  
  localStorage.setItem("theme", newTheme);
});

// ─────────────────────────────────────────────
// ── Mobile sidebar toggle ─────────────────────
// ─────────────────────────────────────────────
menuToggle.addEventListener("click", () => {
  sidebar.classList.toggle("open");
});

// Close sidebar when clicking outside on mobile
document.addEventListener("click", (e) => {
  if (
    sidebar.classList.contains("open") &&
    !sidebar.contains(e.target) &&
    e.target !== menuToggle
  ) {
    sidebar.classList.remove("open");
  }
});

// ─────────────────────────────────────────────
// ── Utility ───────────────────────────────────
// ─────────────────────────────────────────────
function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}
