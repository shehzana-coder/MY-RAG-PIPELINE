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
# Embedding cache
# -------------------------
embedding_cache = {}

# -------------------------
# Weaviate result cache
# -------------------------
weaviate_cache = {}


# -------------------------
# Load environment variables
# -------------------------
load_dotenv()
EMBED_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "openai")
ORG_NAME = os.environ.get("ORG_NAME", "Namal University")
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://127.0.0.1:8081")

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

def get_cached_embedding(text: str, embedder) -> list:
    """
    Returns cached embedding if available; otherwise generates embedding and caches it.
    """
    key = text.strip().lower()  # normalize text for cache

    if key in embedding_cache:
        return embedding_cache[key]

    # Generate embedding
    vector = embedder.embed_query(text)

    # Store in cache
    embedding_cache[key] = vector

    return vector


def get_cache_key(class_name: str, query: str, k: int = 6, alpha: float = 0.5) -> str:
    """
    Generate a unique cache key for a query + class + parameters.
    """
    return f"{class_name}::{query}::k{k}::alpha{alpha}"


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
    """
    Expand the query. 
    NOTE: NLTK expansion disabled as it was causing issues with proper nouns (e.g. Ali -> Cassius Clay).
    Now it just returns the original query.
    """
    return user_query

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
async def hybrid_search_multi_class(query: str, vector: list, k=6, alpha=0.0) -> List[Dict]:
    """
    Perform a single Weaviate hybrid search across all available classes.
    Uses caching to avoid repeated queries.
    """
    # Generate a cache key for this query (all classes)
    key = get_cache_key("ALL_CLASSES", query, k, alpha)
    if key in weaviate_cache:
        return weaviate_cache[key]

    classes = get_available_classes()
    if not classes:
        return []

    # Build GraphQL for all classes
    class_queries = ""
    print(f"  [DEBUG] Querying {len(classes)} classes...")
    for cls in classes:
        class_queries += f"""
        {cls}(
            hybrid: {{
                query: "{query}"
                vector: {vector}
                alpha: {alpha}
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
        """

    gql = f"{{ Get {{ {class_queries} }} }}"
    
    # DEBUG: Print snippet of query to verify vector format/structure
    # print(f"[DEBUG] GraphQL Snippet: {gql[:500]} ... {gql[-200:]}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            r = await client.post(f"{WEAVIATE_URL}/v1/graphql", json={"query": gql})
            r.raise_for_status()
            resp_json = r.json()
            
            if 'errors' in resp_json:
               print(f"[ERROR] GraphQL Errors: {resp_json['errors']}")
            
            data = resp_json.get("data", {}).get("Get", {})
            hits = []
            for cls_name, items in data.items():
                if not items: continue
                for it in items:
                    hits.append({
                        "text": it.get("text", ""),
                        "source": it.get("source", ""),
                        "title": it.get("title", ""),
                        "category": it.get("category", cls_name),
                        "score": it.get("_additional", {}).get("score", 0),
                    })
            # Sort by score descending
            hits.sort(key=lambda x: x.get("score", 0), reverse=True)

            # Store in cache
            weaviate_cache[key] = hits
            return hits[:k*len(classes)]
        except Exception as e:
            print(f"[WARN] Multi-class search failed: {e}")
            return []

# -------------------------
# Decompose Query into Sub-Queries
# -------------------------
def decompose_query(user_query: str) -> List[str]:
    """Decompose a complex query into a list of simple, independent sub-queries."""
    # Heuristic: If query is short or doesn't contain separators, assume it's simple.
    # This avoids unnecessary LLM calls.
    separators = [",", " and ", " or ", ";", "?", "&"]
    is_complex = any(sep in user_query.lower() for sep in separators) and len(user_query.split()) > 3

    if not is_complex:
        return [user_query]

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    
    prompt = (
        f"You are a helpful assistant that splits complex queries into multiple simple sub-queries.\n"
        f"User Query: \"{user_query}\"\n"
        f"Split this query into distinct, standalone sub-queries if it asks about multiple topics.\n"
        f"If the query is already simple, return it as a single-item list.\n"
        f"Return ONLY a JSON list of strings, e.g. [\"Tell me about faculty\", \"Tell me about admission\"]."
    )

    try:
        response = llm.invoke(prompt).content
        response = response.replace("```json", "").replace("```", "").strip()
        sub_queries = json.loads(response)
        
        # Fallback if not a list
        if not isinstance(sub_queries, list):
            return [user_query]
        
        return sub_queries
    except Exception as e:
        print(f"  [WARN] Query decomposition failed: {e}")
        return [user_query]

async def process_single_sub_query(sub_query: str, k: int = 6) -> List[Dict]:
    """Process a single sub-query: Expand -> Identify Classes -> Hybrid Search"""
    # Step 0: Expand query
    expanded_query = expand_query(sub_query)
    print(f"[INFO] Processing Sub-Query: '{sub_query}' -> Expanded: '{expanded_query}'")

    embedder = get_embedder()
    try:
        q_vector = get_cached_embedding(expanded_query, embedder)

    except Exception:
        return []

    all_classes = get_available_classes()
    if not all_classes:
        return []

    # Multi-class search (single call)
    all_hits = await hybrid_search_multi_class(expanded_query, q_vector, k)
    return all_hits

async def retrieve_for_query_async(user_query: str, k: int = 6) -> List[Dict]:
    # Step 1: Decompose Query
    sub_queries = decompose_query(user_query)
    if len(sub_queries) > 1:
        print(f"[INFO] Decomposed into: {sub_queries}")

    # Step 2: Process all sub-queries in parallel
    tasks = [process_single_sub_query(sq, k) for sq in sub_queries]
    results_list = await asyncio.gather(*tasks)

    # Step 3: Aggregate and Deduplicate
    all_hits = []
    seen_hashes = set()

    for hits in results_list:
        for hit in hits:
            # Create a unique hash for deduplication based on text content
            h = hash(hit.get("text", ""))
            if h not in seen_hashes:
                seen_hashes.add(h)
                all_hits.append(hit)

    # Sort by score (descending)
    all_hits.sort(key=lambda x: x.get("score", 0), reverse=True)

    return all_hits[:k * len(sub_queries)]  # Allow more results if multiple queries

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
        print("\n  [RESPONSE]")
        
        full_response = ""
        for chunk in llm.stream([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]):
            content = chunk.content
            print(content, end="", flush=True)
            full_response += content
            
        print("\n") # Newline after streaming
        
        t3 = time.time()
        print(f"  [TIMING] Generation finished in {t3 - t2:.4f}s")
        response = full_response
    except Exception as e:
        return f"I'm sorry, I encountered an error while processing your request. ({str(e)})"

    print(f"  [TIMING] Total Pipeline Time: {time.time() - total_start:.4f}s\n")
    return response
