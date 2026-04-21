"""
Phase 2: Autonomous Content Engine (LangGraph)
State machine: Decide Search → Web Search → Draft Post (JSON output)
"""

import json
import os
import re
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

load_dotenv()

# ── LLM setup ────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.8,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

# ── Mock Search Tool ──────────────────────────────────────────────────────────
MOCK_NEWS = {
    "ai": [
        "OpenAI launches GPT-5 — company claims AGI 'closer than ever'",
        "Google DeepMind publishes AlphaFold 3, accelerating drug discovery",
        "AI coding assistants now write 40 % of all code at major tech firms",
    ],
    "crypto": [
        "Bitcoin hits new all-time high amid regulatory ETF approvals",
        "Ethereum staking yields hit 8 % as DeFi volume explodes",
        "SEC approves first spot-crypto ETF — institutional flood incoming",
    ],
    "tech": [
        "Elon Musk's xAI raises $6B to challenge OpenAI dominance",
        "SpaceX Starship completes first orbital flight successfully",
        "Apple Vision Pro 2 rumoured with 4K micro-OLED displays",
    ],
    "surveillance": [
        "Meta sued in 32 states for harvesting minors' biometric data",
        "EU passes landmark AI Act, banning real-time mass surveillance",
        "Leaked docs reveal government backdoors in major cloud providers",
    ],
    "market": [
        "Fed signals three rate cuts in 2025 — S&P 500 surges 2.4 %",
        "Hedge funds pile into tech as Nasdaq P/E ratios hit 28x",
        "10-year Treasury yield inverts; recession probability climbs to 67 %",
    ],
    "climate": [
        "IPCC warns 1.5 °C threshold breached; tipping points accelerating",
        "Solar installations outpace coal retirements for third straight year",
        "EV adoption plateaus in US amid charging infrastructure gaps",
    ],
}


@tool
def mock_searxng_search(query: str) -> str:
    """Simulates a SearXNG web search. Returns recent mock headlines."""
    query_lower = query.lower()
    for keyword, headlines in MOCK_NEWS.items():
        if keyword in query_lower:
            return "\n".join(f"- {h}" for h in headlines)
    # fallback
    return (
        "- Global uncertainty rises as geopolitical tensions mount\n"
        "- Tech sector faces renewed regulatory scrutiny worldwide"
    )


# ── Graph State ───────────────────────────────────────────────────────────────
class PostState(TypedDict):
    bot_id: str
    bot_persona: str
    search_query: str
    search_results: str
    final_post: dict  # {"bot_id", "topic", "post_content"}


# ── Node 1: Decide Search ─────────────────────────────────────────────────────
def decide_search(state: PostState) -> PostState:
    """LLM picks a topic based on persona and formats a search query."""
    persona = state["bot_persona"]
    prompt = (
        f"You are an AI agent with this persona:\n{persona}\n\n"
        "Decide ONE topic you want to post about today on social media. "
        "Output ONLY a short search query (5 words max) — nothing else."
    )
    response = llm.invoke(prompt)
    query = response.content.strip().strip('"').strip("'")
    print(f"  [Node 1] Search query decided: {query!r}")
    return {**state, "search_query": query}


# ── Node 2: Web Search ────────────────────────────────────────────────────────
def web_search(state: PostState) -> PostState:
    """Runs the mock search tool."""
    results = mock_searxng_search.invoke({"query": state["search_query"]})
    print(f"  [Node 2] Search results:\n{results}")
    return {**state, "search_results": results}


# ── Node 3: Draft Post ────────────────────────────────────────────────────────
def draft_post(state: PostState) -> PostState:
    """LLM writes a 280-char opinionated post. Returns strict JSON."""
    persona = state["bot_persona"]
    results = state["search_results"]
    query = state["search_query"]

    prompt = (
        f"SYSTEM: You are a social media bot. Your persona:\n{persona}\n\n"
        f"CONTEXT (recent news):\n{results}\n\n"
        "TASK: Write one highly opinionated social media post (max 280 characters) "
        f"about the topic: '{query}'.\n"
        "Respond ONLY with a valid JSON object and nothing else:\n"
        '{"bot_id": "<bot_id>", "topic": "<topic>", "post_content": "<post>"}\n'
        f"Use bot_id = '{state['bot_id']}'."
    )

    response = llm.invoke(prompt)
    raw = response.content.strip()

    # strip markdown fences if model adds them
    raw = re.sub(r"```(?:json)?", "", raw).strip("`").strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # last-resort fallback: extract JSON block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(match.group()) if match else {
            "bot_id": state["bot_id"],
            "topic": query,
            "post_content": raw[:280],
        }

    print(f"  [Node 3] Final post JSON:\n{json.dumps(parsed, indent=2)}")
    return {**state, "final_post": parsed}


# ── Build Graph ───────────────────────────────────────────────────────────────
def build_content_graph() -> StateGraph:
    g = StateGraph(PostState)
    g.add_node("decide_search", decide_search)
    g.add_node("web_search", web_search)
    g.add_node("draft_post", draft_post)

    g.set_entry_point("decide_search")
    g.add_edge("decide_search", "web_search")
    g.add_edge("web_search", "draft_post")
    g.add_edge("draft_post", END)
    return g.compile()


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from phase1_router import BOT_PERSONAS

    graph = build_content_graph()

    for bot_id, info in BOT_PERSONAS.items():
        print(f"\n{'='*60}")
        print(f"Running content engine for: {bot_id} ({info['name']})")
        print("=" * 60)

        initial_state: PostState = {
            "bot_id": bot_id,
            "bot_persona": info["description"],
            "search_query": "",
            "search_results": "",
            "final_post": {},
        }

        result = graph.invoke(initial_state)
        print(f"\n✅ Output:\n{json.dumps(result['final_post'], indent=2)}\n")
