"""
Phase 1: Vector-Based Persona Matching (The Router)
Uses FAISS + sentence-transformers for local embedding & cosine similarity.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# ── Bot Personas ──────────────────────────────────────────────────────────────
BOT_PERSONAS = {
    "bot_a": {
        "name": "Tech Maximalist",
        "description": (
            "I believe AI and crypto will solve all human problems. I am highly "
            "optimistic about technology, Elon Musk, and space exploration. "
            "I dismiss regulatory concerns."
        ),
    },
    "bot_b": {
        "name": "Doomer / Skeptic",
        "description": (
            "I believe late-stage capitalism and tech monopolies are destroying "
            "society. I am highly critical of AI, social media, and billionaires. "
            "I value privacy and nature."
        ),
    },
    "bot_c": {
        "name": "Finance Bro",
        "description": (
            "I strictly care about markets, interest rates, trading algorithms, "
            "and making money. I speak in finance jargon and view everything "
            "through the lens of ROI."
        ),
    },
}

# ── Embedding Model ───────────────────────────────────────────────────────────
print("[*] Loading embedding model...")
_model = SentenceTransformer("all-MiniLM-L6-v2")  # fast, small, good quality

# ── Build FAISS Index ─────────────────────────────────────────────────────────
_bot_ids = list(BOT_PERSONAS.keys())
_persona_texts = [BOT_PERSONAS[b]["description"] for b in _bot_ids]

print("[*] Embedding persona descriptions...")
_persona_embeddings = _model.encode(_persona_texts, normalize_embeddings=True)

# Inner-product on L2-normalised vectors == cosine similarity
_dim = _persona_embeddings.shape[1]
_index = faiss.IndexFlatIP(_dim)
_index.add(_persona_embeddings.astype(np.float32))
print(f"[+] FAISS index built: {_index.ntotal} personas stored (dim={_dim})\n")


# ── Public API ────────────────────────────────────────────────────────────────
def route_post_to_bots(post_content: str, threshold: float = 0.40) -> list[dict]:
    """
    Embed *post_content* and return every bot whose persona cosine-similarity
    exceeds *threshold*.

    Returns a list of dicts:
        [{"bot_id": str, "bot_name": str, "similarity": float}, ...]
    """
    post_vec = _model.encode([post_content], normalize_embeddings=True).astype(np.float32)
    similarities, indices = _index.search(post_vec, k=len(_bot_ids))

    matched = []
    for sim, idx in zip(similarities[0], indices[0]):
        if sim >= threshold:
            bot_id = _bot_ids[idx]
            matched.append(
                {
                    "bot_id": bot_id,
                    "bot_name": BOT_PERSONAS[bot_id]["name"],
                    "similarity": round(float(sim), 4),
                }
            )

    # sort by similarity descending
    matched.sort(key=lambda x: x["similarity"], reverse=True)
    return matched


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_posts = [
        "OpenAI just released a new model that might replace junior developers.",
        "Bitcoin hits a new all-time high — hedge funds are going all-in.",
        "Big Tech is surveilling every click you make. This is dystopian.",
        "Fed raises interest rates again, bond yields invert — recession incoming?",
    ]

    for post in test_posts:
        print(f"POST: {post!r}")
        results = route_post_to_bots(post, threshold=0.40)
        if results:
            for r in results:
                print(f"  ✅  {r['bot_id']} ({r['bot_name']})  sim={r['similarity']}")
        else:
            print("  ⚠️  No bot matched above threshold.")
        print()
