import os
import json
import asyncio
import hashlib
import re
import time
from typing import Dict, List, AsyncGenerator
from datetime import datetime
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from dotenv import load_dotenv

from httpx import AsyncClient, Limits, Timeout, AsyncHTTPTransport
import redis.asyncio as redis
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI
import faiss
import weaviate
from weaviate import WeaviateClient
from weaviate.classes.query import HybridFusion, MetadataQuery
from weaviate.collections.classes.filters import Filter
from pyinstrument import Profiler
from circuitbreaker import CircuitBreaker
from prometheus_client import Counter, Histogram, Gauge

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()
EMBED_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "openai")
ORG_NAME = os.environ.get("ORG_NAME", "Namal University")
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://127.0.0.1:8081")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
VLLM_ENDPOINT = os.environ.get("VLLM_ENDPOINT", "http://localhost:8000/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# -------------------------
# Prometheus Metrics
# -------------------------
REQUEST_COUNTER = Counter('query_requests_total', 'Total query requests')
REQUEST_LATENCY = Histogram('query_latency_seconds', 'Query latency in seconds')
CACHE_HITS = Counter('cache_hits_total', 'Total cache hits')
EMBEDDING_TIME = Histogram('embedding_time_seconds', 'Embedding generation time')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active database connections')

# -------------------------
# Connection Manager
# -------------------------
class ConnectionManager:
    """Manage all connections: Weaviate, Redis, HTTP, FAISS, vLLM"""
    
    _weaviate_client = None
    _redis_client = None
    _http_pool = None
    _embedder = None
    _faiss_index = None
    _llm_engine = None
    _executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)

    @classmethod
    async def get_weaviate_client(cls):
        if cls._weaviate_client is None:
            cls._weaviate_client = weaviate.connect_to_local(
                port=8081,
                grpc_port=50051,
                headers={"X-OpenAI-Api-Key": OPENAI_API_KEY}
            )
            ACTIVE_CONNECTIONS.inc()
        return cls._weaviate_client

    @classmethod
    async def get_redis_client(cls):
        if cls._redis_client is None:
            cls._redis_client = redis.from_url(
                REDIS_URL,
                encoding="utf-8",
                decode_responses=False,
                max_connections=100,
                socket_keepalive=True
            )
        return cls._redis_client

    @classmethod
    def get_http_client(cls):
        if cls._http_pool is None:
            transport = AsyncHTTPTransport(
                retries=3,
                limits=Limits(max_keepalive_connections=100, max_connections=1000, keepalive_expiry=60)
            )
            cls._http_pool = AsyncClient(
                timeout=Timeout(10.0, connect=3.0),
                transport=transport,
                follow_redirects=True
            )
        return cls._http_pool

    @classmethod
    def get_embedder(cls):
        if cls._embedder is None:
            if EMBED_PROVIDER == "openai":
                # Prefer faster embedding models for latency
                models_to_try = [
                    "text-embedding-3-small",      # Fastest & cheapest
                    "text-embedding-ada-002",      # Fallback
                    "text-embedding-3-large",      # Best quality (slowest)
                ]
            
            for model in models_to_try:
                try:
                    cls._embedder = OpenAIEmbeddings(
                        model=model,
                        openai_api_key=OPENAI_API_KEY,  # Explicitly pass the key
                        chunk_size=100,
                        show_progress_bar=False
                    )
                    break
                except Exception as e:
                    continue
            
            # If all OpenAI models fail, fall back to HuggingFace
            if cls._embedder is None:
                cls._embedder = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
        return cls._embedder

    @classmethod
    async def get_faiss_index(cls, dimension: int = 384):
        if cls._faiss_index is None:
            cls._faiss_index = faiss.IndexFlatIP(dimension)
            try:
                if os.path.exists("./cache/faiss_index.bin"):
                    cls._faiss_index = faiss.read_index("./cache/faiss_index.bin")
            except Exception:
                pass
        return cls._faiss_index

    @classmethod
    def get_openai_llm(cls):
        if cls._llm_engine is None:
            cls._llm_engine = ChatOpenAI(
                temperature=0.4,
                model="gpt-3.5-turbo",
                openai_api_key=OPENAI_API_KEY,  # Explicitly pass the key
                streaming=True,
                max_tokens=1000
            )
        return cls._llm_engine

    @classmethod
    async def close_all(cls):
        if cls._weaviate_client:
            cls._weaviate_client.close()
            ACTIVE_CONNECTIONS.dec()
        if cls._http_pool:
            await cls._http_pool.aclose()
        if cls._redis_client:
            await cls._redis_client.aclose()
        if cls._executor:
            cls._executor.shutdown(wait=False)
        if cls._faiss_index:
            os.makedirs("./cache", exist_ok=True)
            faiss.write_index(cls._faiss_index, "./cache/faiss_index.bin")

# -------------------------
# Distributed Cache
# -------------------------
class DistributedCache:
    @staticmethod
    async def get(key: str, default=None):
        try:
            client = await ConnectionManager.get_redis_client()
            value = await client.get(f"rag:{key}")
            if value:
                CACHE_HITS.inc()
                return json.loads(value)
        except Exception as e:
            # Silently fail on cache errors
            pass
        return default

    @staticmethod
    async def set(key: str, value, ttl: int = 3600):
        try:
            client = await ConnectionManager.get_redis_client()
            await client.setex(f"rag:{key}", ttl, json.dumps(value, default=str))
        except Exception as e:
            # Silently fail on cache errors
            pass

    @staticmethod
    async def batch_get(keys: List[str]):
        try:
            client = await ConnectionManager.get_redis_client()
            prefixed_keys = [f"rag:{k}" for k in keys]
            values = await client.mget(prefixed_keys)
            return [json.loads(v) if v else None for v in values]
        except Exception as e:
            # Silently fail on cache errors
            return [None] * len(keys)

    @staticmethod
    async def clear_pattern(pattern: str = "rag:*"):
        try:
            client = await ConnectionManager.get_redis_client()
            keys = await client.keys(pattern)
            if keys:
                await client.delete(*keys)
        except Exception as e:
            print(f"[CACHE CLEAR ERROR] {e}")

# -------------------------
# Embedding Service
# -------------------------
class FastEmbeddingService:
    @staticmethod
    @lru_cache(maxsize=10000)
    def _get_memory_cache_key(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:16]

    @staticmethod
    async def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
        start_time = datetime.now()
        cache_keys = [f"emb:{hashlib.md5(t.encode()).hexdigest()}" for t in texts]
        cached_results = await DistributedCache.batch_get(cache_keys)

        embeddings = []
        uncached_texts = []
        uncached_indices = []

        for i, (text, cached) in enumerate(zip(texts, cached_results)):
            if cached:
                embeddings.append(cached)
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            embedder = ConnectionManager.get_embedder()
            loop = asyncio.get_event_loop()
            new_embeddings = await loop.run_in_executor(ConnectionManager._executor,
                                                        lambda: embedder.embed_documents(uncached_texts))
            for idx, emb in zip(uncached_indices, new_embeddings):
                embeddings[idx] = emb
            # Cache new embeddings
            cache_tasks = [DistributedCache.set(cache_keys[text_idx], new_embeddings[idx], ttl=86400)
                           for idx, text_idx in enumerate(uncached_indices)]
            await asyncio.gather(*cache_tasks, return_exceptions=True)
            # Add to FAISS index
            try:
                faiss_index = await ConnectionManager.get_faiss_index(len(new_embeddings[0]))
                vectors = np.array(new_embeddings).astype('float32')
                faiss.normalize_L2(vectors)
                faiss_index.add(vectors)
            except Exception as e:
                print(f"[FAISS ERROR] {e}")

        EMBEDDING_TIME.observe((datetime.now() - start_time).total_seconds())
        return embeddings

    @staticmethod
    async def find_similar_cached(query_embedding: List[float], k: int = 3) -> List[Dict]:
        try:
            faiss_index = await ConnectionManager.get_faiss_index(len(query_embedding))
            query_vector = np.array([query_embedding]).astype('float32')
            faiss.normalize_L2(query_vector)
            distances, indices = faiss_index.search(query_vector, k)

            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx != -1 and distance > 0.7:
                    cached = await DistributedCache.get(f"vector:{idx}")
                    if cached:
                        cached['similarity'] = float(distance)
                        results.append(cached)
            return results
        except Exception as e:
            print(f"[SIMILARITY SEARCH ERROR] {e}")
            return []

# -------------------------
# Weaviate Hybrid Search
# -------------------------
class FastWeaviateSearch:
    @staticmethod
    @CircuitBreaker(failure_threshold=5, recovery_timeout=60)
    async def hybrid_search(query: str, query_vector: List[float], class_names: List[str], k: int = 6, alpha: float = 0.5) -> List[Dict]:
        cache_key = f"search::{hashlib.md5(query.encode()).hexdigest()}::{'-'.join(sorted(class_names))}::{k}::{alpha}"
        cached = await DistributedCache.get(cache_key)
        if cached:
            return cached

        similar_results = await FastEmbeddingService.find_similar_cached(query_vector, k=2)

        client = await ConnectionManager.get_weaviate_client()
        all_results = []
        search_tasks = [FastWeaviateSearch._search_single_class(client, cls, query, query_vector, k, alpha) for cls in class_names]

        try:
            results = await asyncio.wait_for(asyncio.gather(*search_tasks, return_exceptions=True), timeout=5.0)
        except asyncio.TimeoutError:
            results = []

        for result in results:
            if isinstance(result, Exception):
                continue
            all_results.extend(result)
        all_results.extend(similar_results)

        seen_texts = set()
        unique_results = []
        for r in sorted(all_results, key=lambda x: x.get('score', 0), reverse=True):
            text = r.get('text', '')
            if text and text not in seen_texts:
                seen_texts.add(text)
                unique_results.append(r)

        final_results = unique_results[:k]
        asyncio.create_task(DistributedCache.set(cache_key, final_results, ttl=300))
        return final_results

    @staticmethod
    async def _search_single_class(client: WeaviateClient, class_name: str, query: str, vector: List[float], k: int, alpha: float) -> List[Dict]:
        try:
            collection = client.collections.get(class_name)
            # Weaviate v4 hybrid search is synchronous, not async
            results = collection.query.hybrid(
                query=query,
                vector=vector,
                limit=k * 2,
                alpha=alpha,
                fusion_type=HybridFusion.RELATIVE_SCORE,
                return_metadata=MetadataQuery(score=True, distance=True, certainty=True)
            )
            formatted = []
            for obj in results.objects[:k]:
                formatted.append({
                    'text': obj.properties.get('text', ''),
                    'source': obj.properties.get('source', 'N/A'),
                    'title': obj.properties.get('title', 'N/A'),
                    'category': obj.properties.get('category', class_name),
                    'score': obj.metadata.score if obj.metadata else 0.0,
                    'certainty': obj.metadata.certainty if obj.metadata else 0.0,
                    'class': class_name
                })
            return formatted
        except Exception as e:
            return []

# -------------------------
# Query Processor with vLLM
# -------------------------
class FastQueryProcessor:
    @staticmethod
    async def decompose_query(query: str) -> List[str]:
        # OPTIMIZATION: Skip decomposition for simple queries (single topic, short)
        # This saves ~3s of LLM latency
        word_count = len(query.split())
        query_lower = query.lower()
        
        # Simple heuristic: if <12 words and no "and", "or", "," then it's a simple query
        is_simple = (word_count < 12 and 
                    " and " not in query_lower and 
                    " or " not in query_lower and 
                    "," not in query_lower)
        
        if is_simple:
            print(f"[DEBUG] Skipped decomposition - simple query detected")
            return [query]
        
        cache_key = f"decompose::{hashlib.md5(query.encode()).hexdigest()}"
        cached = await DistributedCache.get(cache_key)
        if cached:
            return cached

        try:
            llm = ConnectionManager.get_openai_llm()
            prompt = f"""Decompose this query into sub-queries if it contains multiple topics.
                Query: {query}

                Return ONLY a JSON array with no explanation, like this:
                ["subquery1", "subquery2"]

                If it's a single topic, return:
                ["{query}"]

                JSON array:"""
            
            response = llm.invoke([{"role": "user", "content": prompt}])
            response_text = response.content.strip()
            
            # Clean up the response more aggressively
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            # Extract JSON array if wrapped in text
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
            
            subqueries = json.loads(response_text)
            if not isinstance(subqueries, list):
                subqueries = [query]
            
            asyncio.create_task(DistributedCache.set(cache_key, subqueries, ttl=86400))  # Cache 24 hours
            return subqueries
            
        except Exception as e:
            print(f"[DECOMPOSE ERROR] {e}")
            # Better fallback - just return the original query
            return [query]
    @staticmethod
    async def select_relevant_classes(query: str, all_classes: List[str]) -> List[str]:
        """Use LLM to intelligently select 5-10 most relevant classes for the query"""
        
        word_count = len(query.split())
        is_simple = word_count < 12
        
        if is_simple:
            # Use Weaviate's hybrid search for simple queries
            try:
                client = await ConnectionManager.get_weaviate_client()
                relevant_classes = []
                scores = {}
                
                # Query each collection with hybrid search to determine relevance
                for class_name in all_classes:
                    try:
                        collection = client.collections.get(class_name)
                        
                        # Perform hybrid search
                        # alpha=0.6 slightly favors vector search for semantic understanding
                        response = collection.query.hybrid(
                            query=query,
                            limit=1,
                            alpha=0.6,
                            return_metadata=['score']
                        )
                        
                        # If we get results, this collection is relevant
                        if response.objects:
                            score = response.objects[0].metadata.score
                            scores[class_name] = score
                            
                    except Exception:
                        continue
                
                # Sort by score and return top 3 classes
                sorted_classes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                relevant_classes = [c[0] for c in sorted_classes[:3]]
                
                if relevant_classes:
                    return relevant_classes
                    
            except Exception:
                pass
        
        # For complex queries, use LLM-based selection with caching
        cache_key = f"classes::{hashlib.md5(query.encode()).hexdigest()}"
        cached = await DistributedCache.get(cache_key)
        if cached:
            return cached
        
        try:
            llm = ConnectionManager.get_openai_llm()
            classes_str = ", ".join(all_classes)
            prompt = f"""Select only the 2 to 4 most relevant database class for this query.
                Classes: {classes_str}
                Query: "{query}"
                Return JSON array only: ["Class1"]"""
            
            response = llm.invoke([{"role": "user", "content": prompt}])
            response_text = response.content.strip()
            
            # Clean up the response
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            # Extract JSON array
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
            
            selected_classes = json.loads(response_text)
            
            # Ensure returned classes actually exist
            valid_classes = [c for c in selected_classes if c in all_classes]
            
            # Ensure we have at least some classes
            if not valid_classes:
                valid_classes = all_classes[:8]
            
            asyncio.create_task(DistributedCache.set(cache_key, valid_classes, ttl=86400))
            return valid_classes
            
        except Exception as e:
            # Fallback: return first 8 classes
            return all_classes[:8]

    @staticmethod
    async def expand_query(query: str) -> str:
        """Use LLM to expand/clarify the query for better retrieval"""
        cache_key = f"expand::{hashlib.md5(query.encode()).hexdigest()}"
        cached = await DistributedCache.get(cache_key)
        if cached:
            return cached

        try:
            llm = ConnectionManager.get_openai_llm()
            prompt = f"""Expand query with synonyms and related terms. Keep it concise (max 15 words).
Query: "{query}"
Expanded:"""
            
            response = llm.invoke([{"role": "user", "content": prompt}])
            expanded = response.content.strip()
            
            asyncio.create_task(DistributedCache.set(cache_key, expanded, ttl=86400))  # Cache 24 hours
            return expanded
            
        except Exception as e:
            # Fallback: return original query
            return query

# -------------------------
# LLM Generator
# -------------------------
class FastLLMGenerator:
    @staticmethod
    async def generate_response(query: str, context: str, stream: bool = False) -> AsyncGenerator[str, None] | str:
        cache_key = f"response::{hashlib.md5((query + context[:500]).encode()).hexdigest()}"
        if not stream:
            cached = await DistributedCache.get(cache_key)
            if cached:
                return cached
        prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context.

            Context:
            {context}

            Question: {query}

            Instructions:
            - Answer ONLY using information from the context above
            - Use ALL relevant information from the context to provide a COMPREHENSIVE answer
            - If the context doesn't contain relevant information to answer the question, respond with: "I don't have information about this in my knowledge base."
            - Do NOT use your general knowledge
            - Be detailed and thorough - include all relevant details, names, and information from the context
            - Cite sources when possible

            Answer:"""
        try:
            if stream:
                response = await FastLLMGenerator._stream_response(prompt)
                asyncio.create_task(DistributedCache.set(cache_key, response, ttl=1800))
                return response
            else:
                response = await FastLLMGenerator._generate_full_response(prompt)
                asyncio.create_task(DistributedCache.set(cache_key, response, ttl=1800))
                return response
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if stream:
                return error_msg
            else:
                return error_msg

    @staticmethod
    async def _stream_response(prompt: str) -> AsyncGenerator[str, None]:
        """Stream response using OpenAI's native streaming"""
        try:
            llm = ConnectionManager.get_openai_llm()
            
            # Use astream for native async streaming
            async for chunk in llm.astream([{"role": "user", "content": prompt}]):
                if hasattr(chunk, 'content'):
                    yield chunk.content
                    
        except Exception as e:
            yield f"Error: {str(e)}"
        

    @staticmethod
    async def _generate_full_response(prompt: str) -> str:
        try:
            llm = ConnectionManager.get_openai_llm()
            response = llm.invoke([prompt])
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"  

# -------------------------
# RAG Processor
# -------------------------
class FastRAGProcessor:
    @staticmethod
    async def process_query(query: str, k: int = 15, stream: bool = False) -> AsyncGenerator[str, None] | str:
        REQUEST_COUNTER.inc()
        with REQUEST_LATENCY.time():
            profiler = Profiler(async_mode='enabled')
            profiler.start()
            try:
                # 1. Decompose query into sub-queries (cached for 24 hours)
                subqueries = await FastQueryProcessor.decompose_query(query)
                
                # 2. Only expand query for complex multi-part questions (skip for simple queries)
                is_complex_query = len(subqueries) > 1 or len(query.split()) > 15
                expanded_query = await FastQueryProcessor.expand_query(query) if is_complex_query else query
                
                # 3. Use LLM to select relevant classes (run in parallel with embeddings)
                client = await ConnectionManager.get_weaviate_client()
                all_classes = list(client.collections.list_all().keys())
                
                # 4. Run class selection and embedding in parallel for speed
                class_selection_task = FastQueryProcessor.select_relevant_classes(expanded_query, all_classes)
                embedding_task = FastEmbeddingService.get_embeddings_batch(subqueries)
                
                relevant_classes, embeddings_list = await asyncio.gather(
                    class_selection_task,
                    embedding_task
                )
                
                # 5. Search in selected classes only - retrieve more results for comprehensive context
                # Use higher k for better coverage (retrieves more context per class)
                search_k = max(6, k)  # Retrieve 12 results per class for comprehensive context (balanced for latency)
                search_tasks = []
                for subquery, embedding in zip(subqueries, embeddings_list):
                    if embedding:
                        search_tasks.append(FastWeaviateSearch.hybrid_search(subquery, embedding, relevant_classes, k=search_k))

                search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
                all_hits = []
                seen_texts = set()
                for results in search_results:
                    if isinstance(results, Exception):
                        continue
                    for hit in results:
                        text = hit.get('text', '')
                        if text and text not in seen_texts:
                            seen_texts.add(text)
                            all_hits.append(hit)

                all_hits.sort(key=lambda x: x.get('score', 0), reverse=True)
                
                # ✅ ADAPTIVE RELEVANCE THRESHOLD - Lower threshold for comprehensive context retrieval
                RELEVANCE_THRESHOLD = 0.6  # More lenient threshold for better coverage
                relevant_hits = [hit for hit in all_hits if hit.get('score', 0) >= RELEVANCE_THRESHOLD]
                
                # ✅ FALLBACK: If insufficient results, include lower-scored hits for comprehensive coverage
                if len(relevant_hits) < 5:
                    # Include all hits if we have too few high-confidence results
                    relevant_hits = all_hits[:9] if all_hits else []
                else:
                    # Limit to top 12 for balance between comprehensiveness and latency
                    relevant_hits = relevant_hits[:5]
                
                # ✅ IF STILL NO CONTEXT, RETURN NO RESULTS MESSAGE
                if not relevant_hits:
                    no_context_msg = (
                        "I don't have information about this topic in my knowledge base. "
                        "Please try a different question or visit our website for more information."
                    )
                    return no_context_msg
                
                # Use ALL relevant hits (not limited by k) for comprehensive context
                # This ensures the LLM has full information to generate complete responses
                context = "\n\n".join([f"[Source: {hit.get('title', hit.get('source', 'Unknown'))}]\n{hit.get('text', '')}" for hit in relevant_hits])
                print(f"[DEBUG] Retrieved {len(relevant_hits)} context pieces ({len(context)} chars total)")

                if stream:
                    response = await FastLLMGenerator.generate_response(query, context, stream=True)
                    return response
                else:
                    response = await FastLLMGenerator.generate_response(query, context, stream=False)
                    return response  
            finally:
                profiler.stop()
                # profiler.print()  # Disabled for cleaner output


# -------------------------
# Public API - Synchronous Wrapper
# -------------------------
def perform_query(query: str) -> str:
    """Synchronous wrapper to run async query processing"""
    return asyncio.run(FastRAGProcessor.process_query(query, stream=False))
