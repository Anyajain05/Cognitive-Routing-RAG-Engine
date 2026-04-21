"""
main.py — Runs all three phases of the Grid07 AI assignment end-to-end.
Captures output to logs/execution_log.txt automatically.
"""

import json
import sys
import os

# ── Logging: tee stdout to file ───────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
log_file = open("logs/execution_log.txt", "w", encoding="utf-8")


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except (ValueError, OSError):
                pass
    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except (ValueError, OSError):
                pass


sys.stdout = Tee(sys.stdout, log_file)

# ── Phase 1 ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PHASE 1 — Vector-Based Persona Routing")
print("=" * 65)

from phase1_router import route_post_to_bots, BOT_PERSONAS

test_posts = [
    "OpenAI just released a new model that might replace junior developers.",
    "Bitcoin hits a new all-time high — hedge funds are going all-in.",
    "Big Tech is surveilling every click you make. This is dystopian.",
    "Fed raises interest rates again, bond yields invert — recession incoming?",
]

for post in test_posts:
    print(f"\nPOST: {post!r}")
    results = route_post_to_bots(post, threshold=0.40)
    if results:
        for r in results:
            print(f"  ✅  {r['bot_id']:6s} ({r['bot_name']})  similarity={r['similarity']}")
    else:
        print("  ⚠️   No bot matched above threshold.")

# ── Phase 2 ───────────────────────────────────────────────────────────────────
print("\n\n" + "=" * 65)
print("  PHASE 2 — Autonomous Content Engine (LangGraph)")
print("=" * 65)

from phase2_content_engine import build_content_graph, PostState

graph = build_content_graph()
phase2_outputs = []

for bot_id, info in BOT_PERSONAS.items():
    print(f"\n--- Running graph for {bot_id} ({info['name']}) ---")
    initial_state: PostState = {
        "bot_id": bot_id,
        "bot_persona": info["description"],
        "search_query": "",
        "search_results": "",
        "final_post": {},
    }
    result = graph.invoke(initial_state)
    phase2_outputs.append(result["final_post"])
    print(f"\n✅ JSON Output:\n{json.dumps(result['final_post'], indent=2)}")

# ── Phase 3 ───────────────────────────────────────────────────────────────────
print("\n\n" + "=" * 65)
print("  PHASE 3 — Combat Engine (Deep Thread RAG + Injection Defense)")
print("=" * 65)

from phase3_combat_engine import (
    generate_defense_reply,
    PARENT_POST,
    COMMENT_HISTORY,
    HUMAN_REPLY_NORMAL,
    HUMAN_REPLY_INJECTION,
)

bot_persona = BOT_PERSONAS["bot_a"]["description"]

print("\n--- Scenario A: Normal Adversarial Reply ---")
print(f"Human: {HUMAN_REPLY_NORMAL!r}")
reply_a = generate_defense_reply(bot_persona, PARENT_POST, COMMENT_HISTORY, HUMAN_REPLY_NORMAL)
print(f"Bot A: {reply_a}")

print("\n--- Scenario B: Prompt Injection Attack ---")
print(f"Human: {HUMAN_REPLY_INJECTION!r}")
reply_b = generate_defense_reply(bot_persona, PARENT_POST, COMMENT_HISTORY, HUMAN_REPLY_INJECTION)
print(f"Bot A (should stay in persona): {reply_b}")

print("\n✅ All three phases complete. Log saved to logs/execution_log.txt")
log_file.close()
