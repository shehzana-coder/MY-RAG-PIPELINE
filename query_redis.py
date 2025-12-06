import os
import json
import asyncio
import redis
import numpy as np
from redis.commands.search.query import Query
from dotenv import load_dotenv
from typing import List, Dict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import nltk
from nltk.corpus import wordnet
import re

# Load env
load_dotenv()
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
ORG_NAME = os.environ.get("ORG_NAME", "Namal University")
INDEX_NAME = "namal_idx"

# Cache
embedding_cache = {}

def get_redis_client():
    return redis.from_url(REDIS_URL, decode_responses=True)

def get_embedder():
    return OpenAIEmbeddings()

def get_cached_embedding(text: str, embedder) -> list:
    key = text.strip().lower()
    if key in embedding_cache:
        return embedding_cache[key]
    vector = embedder.embed_query(text)
    embedding_cache[key] = vector
    return vector

# -------------------------
# Query Expansion
# -------------------------
def expand_query(user_query: str, max_synonyms_per_word=2) -> str:
    # Use existing logic from query.py
    try:
        nltk.data.find('corpora/wordnet.zip')
    except LookupError:
         nltk.download('wordnet')

    sub_queries = re.split(r'[?;.]', user_query)
    sub_queries = [q.strip() for q in sub_queries if q.strip()]

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
    
    return ' '.join(expanded_queries)

# -------------------------
# Decompose Query
# -------------------------
def decompose_query(user_query: str) -> List[str]:
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
        if not isinstance(sub_queries, list):
            return [user_query]
        return sub_queries
    except Exception as e:
        print(f"  [WARN] Query decomposition failed: {e}")
        return [user_query]

# -------------------------
# Search Logic
# -------------------------
def search_redis(query_text: str, vector: list, k: int = 6) -> List[Dict]:
    r = get_redis_client()
    
    # Prepare Vector for Redis (bytes)
    vector_bytes = np.array(vector, dtype=np.float32).tobytes()

    # Query: Vector Similarity (KNN)
    # Syntax: "*=>[KNN 6 @vector $vec_param AS vector_score]"
    q = Query(f"*=>[KNN {k} @vector $vec_param AS vector_score]")\
        .sort_by("vector_score")\
        .return_fields("text", "source", "title", "category", "vector_score")\
        .dialect(2)
    
    params = {"vec_param": vector_bytes}
    
    try:
        res = r.ft(INDEX_NAME).search(q, query_params=params)
        hits = []
        for doc in res.docs:
            score = 1 - float(doc.vector_score) # Convert distance to similarity score approx
            hits.append({
                "text": doc.text,
                "source": doc.source,
                "title": doc.title,
                "category": doc.category,
                "score": score
            })
        return hits
    except Exception as e:
        print(f"[WARN] Redis search failed: {e}")
        return []

async def process_single_sub_query(sub_query: str, k: int = 6) -> List[Dict]:
    expanded_query = expand_query(sub_query)
    print(f"[INFO] Processing Sub-Query: '{sub_query}' -> Expanded: '{expanded_query}'")
    
    embedder = get_embedder()
    try:
        q_vector = get_cached_embedding(expanded_query, embedder)
    except Exception:
        return []
        
    return search_redis(expanded_query, q_vector, k)

async def retrieve_for_query_async(user_query: str, k: int = 6) -> List[Dict]:
    sub_queries = decompose_query(user_query)
    if len(sub_queries) > 1:
        print(f"[INFO] Decomposed into: {sub_queries}")

    tasks = [process_single_sub_query(sq, k) for sq in sub_queries]
    results_list = await asyncio.gather(*tasks)

    all_hits = []
    seen_hashes = set()

    for hits in results_list:
        for hit in hits:
            h = hash(hit.get("text", ""))
            if h not in seen_hashes:
                seen_hashes.add(h)
                all_hits.append(hit)

    all_hits.sort(key=lambda x: x.get("score", 0), reverse=True)
    return all_hits[:k * len(sub_queries)]

def retrieve_for_query(user_query: str, k: int = 6) -> List[Dict]:
    return asyncio.run(retrieve_for_query_async(user_query, k))

# -------------------------
# Generation
# -------------------------
def perform_query(user_query: str) -> str:
    import time
    total_start = time.time()

    print("\n  [TIMING] Starting Retrieval (Redis)...")
    t0 = time.time()
    hits = retrieve_for_query(user_query, k=6)
    t1 = time.time()
    print(f"  [TIMING] Retrieval finished in {t1 - t0:.4f}s")

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
        print("\n")
        t3 = time.time()
        print(f"  [TIMING] Generation finished in {t3 - t2:.4f}s")
        response = full_response
    except Exception as e:
        return f"I'm sorry, I encountered an error. ({str(e)})"

    print(f"  [TIMING] Total Pipeline Time: {time.time() - total_start:.4f}s\n")
    return response
