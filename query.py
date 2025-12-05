import os
import json
import asyncio
from dotenv import load_dotenv
import httpx
from typing import Dict, List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import wordnet
import re


# -------------------------
# Load environment variables
# -------------------------
load_dotenv()
EMBED_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "openai")
ORG_NAME = os.environ.get("ORG_NAME", "Namal University")
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8081")

# -------------------------
# Embedding & Schema Caching
# -------------------------
def get_embedder():
    if EMBED_PROVIDER == "openai":
        return OpenAIEmbeddings()
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        model = os.environ.get("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name=model)

CACHE_FILE = os.path.join(os.path.dirname(__file__), ".weaviate_classes_cache.json")

def get_available_classes() -> List[str]:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                print("  [INFO] Loading classes from cache...")
                return json.load(f)
        except Exception:
            pass

    try:
        print("  [INFO] Fetching classes from Weaviate schema...")
        r = httpx.get(f"{WEAVIATE_URL}/v1/schema", timeout=2)
        r.raise_for_status()
        schema = r.json()
        classes = [c.get("class") for c in schema.get("classes", [])]

        try:
            with open(CACHE_FILE, 'w') as f:
                json.dump(classes, f)
        except Exception as e:
            print(f"  [WARN] Failed to write cache: {e}")

        return classes
    except Exception:
        return []

# -------------------------
# Query Expansion using Synonyms
# -------------------------
def expand_query(user_query: str, max_synonyms_per_word=2) -> str:
    """Expand the query by adding synonyms and handle multiple queries in one input."""
    # Split user query into sub-queries using ?, ;, .
    sub_queries = re.split(r'[?;.]', user_query)
    sub_queries = [q.strip() for q in sub_queries if q.strip()]  # remove empty

    expanded_queries = []

    for sub_query in sub_queries:
        tokens = re.findall(r'\w+', sub_query.lower())
        expanded_terms = set(tokens)

        for token in tokens:
            synonyms = set()
            for syn in wordnet.synsets(token):
                for lemma in syn.lemmas():
                    lemma_name = lemma.name().replace('_', ' ')
                    if lemma_name != token:
                        synonyms.add(lemma_name)
            synonyms = list(synonyms)[:max_synonyms_per_word]
            expanded_terms.update(synonyms)

        expanded_queries.append(' '.join(expanded_terms))

    # Combine all expanded sub-queries into one string
    expanded_query = ' '.join(expanded_queries)
    return expanded_query

# -------------------------
# Smart Class Selection using LLM
# -------------------------
def identify_relevant_classes(user_query: str, available_classes: List[str]) -> List[str]:
    if not available_classes:
        return []

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    classes_str = ", ".join(available_classes)

    prompt = (
        f"You are a smart classifier for a vector database.\n"
        f"Available Classes (Collections): {classes_str}\n\n"
        f"User Query: \"{user_query}\"\n\n"
        f"Task: Identify which 1 to 3 classes from the list above are most likely to contain the answer.\n"
        f"Return ONLY a JSON list of strings, e.g. [\"Faculty\", \"AdmissionCriteria\"].\n"
        f"If unsure, select the 'General' class or the most broad ones."
    )

    try:
        response = llm.invoke(prompt).content
        response = response.replace("```json", "").replace("```", "").strip()
        selected_classes = json.loads(response)
        valid_classes = [c for c in selected_classes if c in available_classes]

        if not valid_classes:
            print("  [WARN] Router returned no valid classes. Defaulting to first 3 classes.")
            return available_classes[:3]
        return valid_classes
    except Exception as e:
        print(f"  [ERROR] Class selection failed: {e}")
        return available_classes[:3]

# -------------------------
# Async Hybrid Search (Parallel HTTP)
# -------------------------
async def hybrid_search_async(class_name: str, query: str, vector: list, k=6) -> List[Dict]:
    gql = f'''
    {{
      Get {{

        {class_name}(
          hybrid: {{
            query: "{query}"
            vector: {vector}
            alpha: 0.5
          }}
          limit: {k}
        ) {{
          text
          source
          title
          category
          _additional {{
            score
            distance
          }}
        }}
      }}
    }}
    '''
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            r = await client.post(f"{WEAVIATE_URL}/v1/graphql", json={"query": gql})
            if r.status_code == 200:
                data = r.json()
                items = data.get("data", {}).get("Get", {}).get(class_name, [])
                hits = []
                for it in items:
                    hits.append({
                        "text": it.get("text", ""),
                        "source": it.get("source", ""),
                        "title": it.get("title", ""),
                        "category": it.get("category", class_name),
                        "score": it.get("_additional", {}).get("score", 0),
                    })
                return hits
        except Exception as e:
            print(f"[WARN] Search failed for {class_name}: {e}")
            return []
    return []

async def retrieve_for_query_async(user_query: str, k: int = 6) -> List[Dict]:
    # Step 0: Expand query
    expanded_query = expand_query(user_query)
    print(f"[INFO] Expanded Query: {expanded_query}")

    embedder = get_embedder()
    try:
        q_vector = embedder.embed_query(expanded_query)
    except Exception:
        return []

    all_classes = get_available_classes()
    if not all_classes:
        return []

    target_classes = identify_relevant_classes(user_query, all_classes)
    print(f"[INFO] Searching in classes: {target_classes}")

    tasks = [hybrid_search_async(cls, expanded_query, q_vector, k) for cls in target_classes]
    results = await asyncio.gather(*tasks)

    all_hits = [hit for class_hits in results for hit in class_hits]
    all_hits.sort(key=lambda x: x.get("score", 0), reverse=True)

    return all_hits[:k]

def retrieve_for_query(user_query: str, k: int = 6) -> List[Dict]:
    return asyncio.run(retrieve_for_query_async(user_query, k))

# -------------------------
# LLM Answer Generation
# -------------------------
def perform_query(user_query: str) -> str:
    import time
    total_start = time.time()

    # 1. Retrieve
    print("\n  [TIMING] Starting Retrieval...")
    t0 = time.time()
    hits = retrieve_for_query(user_query, k=6)
    t1 = time.time()
    print(f"  [TIMING] Retrieval finished in {t1 - t0:.4f}s")

    # 2. Prepare Context
    if not hits:
        context = "No specific documents found."
    else:
        context_parts = []
        for h in hits:
            src = h.get("title") or h.get("source") or "Unknown"
            cat = h.get("category", "General")
            txt = h.get("text", "").strip()
            context_parts.append(f"[Source: {src} | Category: {cat}]\n{txt}")
        context = "\n\n---\n\n".join(context_parts)

    # 3. Generate Answer
    print("  [TIMING] Starting Generation...")
    llm = ChatOpenAI(temperature=0.4, model="gpt-3.5-turbo")
    system_prompt = (
        f"You are a friendly and professional AI assistant for {ORG_NAME}. "
        "Answer ONLY using the context provided.\n"
    )
    user_prompt = f"Context:\n{context}\n\nUser Question: {user_query}\nAnswer:"

    try:
        t2 = time.time()
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]).content
        t3 = time.time()
        print(f"  [TIMING] Generation finished in {t3 - t2:.4f}s")
    except Exception as e:
        return f"I'm sorry, I encountered an error while processing your request. ({str(e)})"

    print(f"  [TIMING] Total Pipeline Time: {time.time() - total_start:.4f}s\n")
    return response
