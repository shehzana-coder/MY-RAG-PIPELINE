import os
from dotenv import load_dotenv
import requests
from typing import Dict, List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()
EMBED_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "openai")
ORG_NAME = os.environ.get("ORG_NAME", "Namal University")
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8081")

def get_embedder():
    if EMBED_PROVIDER == "openai":
        return OpenAIEmbeddings()
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        model = os.environ.get("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name=model)

def get_available_classes() -> List[str]:
    try:
        r = requests.get(f"{WEAVIATE_URL}/v1/schema", timeout=2)
        r.raise_for_status()
        schema = r.json()
        return [c.get("class") for c in schema.get("classes", [])]
    except Exception:
        return []

def retrieve_for_query(user_query: str, k: int = 6) -> List[Dict]:
    """Embed and retrieve top documents across all classes."""
    embedder = get_embedder()
    try:
        q_vector = embedder.embed_query(user_query)
    except Exception:
        return []

    classes = get_available_classes()
    if not classes:
        return []

    all_hits = []
    # Search all classes (since we have multiple categories as classes)
    for class_name in classes:
        gql = f'''{{ Get {{ {class_name}(nearVector: {{ vector: {q_vector} }} limit: {k}) {{ text source title category _additional {{ distance }} }} }} }}'''
        try:
            r = requests.post(f"{WEAVIATE_URL}/v1/graphql", json={"query": gql}, timeout=5)
            if r.status_code == 200:
                data = r.json()
                items = data.get("data", {}).get("Get", {}).get(class_name, [])
                if items:
                    for it in items:
                        all_hits.append({
                            "text": it.get("text", ""),
                            "source": it.get("source", ""),
                            "title": it.get("title", ""),
                            "category": it.get("category", class_name),
                            "distance": it.get("_additional", {}).get("distance", 0),
                        })
        except Exception:
            continue
            
    # Sort by distance (ascending) and take top k
    all_hits.sort(key=lambda x: x.get("distance", 0))
    return all_hits[:k]

def perform_query(user_query: str) -> str:
    """
    Simplified query pipeline:
    1. Retrieve relevant docs.
    2. Generate friendly answer.
    """
    # 1. Retrieve
    hits = retrieve_for_query(user_query, k=6)
    
    # 2. Prepare Context
    if not hits:
        context = "No specific documents found."
    else:
        context_parts = []
        for h in hits:
            src = h.get("title") or h.get("source") or "Unknown"
            txt = h.get("text", "").strip()
            context_parts.append(f"[Source: {src}]\n{txt}")
        context = "\n\n---\n\n".join(context_parts)

    # 3. Generate Answer
    llm = ChatOpenAI(temperature=0.4, model="gpt-3.5-turbo") # Slightly higher temp for friendliness
    
    system_prompt = (
        f"You are a friendly and professional AI assistant for {ORG_NAME}. "
        "Your goal is to help users find information about the university, faculty, and programs.\n\n"
        "Guidelines:\n"
        "- Answer the user's question using ONLY the provided context.\n"
        "- If the context doesn't have the answer, politely say you don't have that information right now.\n"
        "- Be warm, engaging, and helpful (like a human talking to a human).\n"
        "- Format your answer clearly (use bullet points if listing items).\n"
        "- Do not mention 'context' or 'retrieved documents' to the user. Just answer naturally."
    )
    
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"User Question: {user_query}\n\n"
        "Answer:"
    )

    try:
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]).content
    except Exception as e:
        return f"I'm sorry, I encountered an error while processing your request. ({str(e)})"

    return response
