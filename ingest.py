import os
import json
import time
from typing import List, Dict
from dotenv import load_dotenv
from tqdm import tqdm
import requests
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8081")
EMBED_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "openai")
# Path to the JSON file
JSON_DATA_PATH = os.path.join(os.path.dirname(__file__), "weaviate_ready_data", "faculty.json")

def get_embedder():
    if EMBED_PROVIDER == "openai":
        return OpenAIEmbeddings()
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        model = os.environ.get("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name=model)

def ingest():
    print(f"\n{'='*70}")
    print(f"  Weaviate Ingestion (Optimized)")
    print(f"{'='*70}")
    print(f"  Target: {WEAVIATE_URL}")
    print(f"  Source: {JSON_DATA_PATH}")

    if not os.path.exists(JSON_DATA_PATH):
        print(f"  [ERROR] File not found: {JSON_DATA_PATH}")
        return

    print(f"  [INFO] Loading JSON data...")
    with open(JSON_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  [INFO] Loaded {len(data)} records.")

    embedder = get_embedder()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    batch_endpoint = f"{WEAVIATE_URL}/v1/batch/objects"
    
    all_chunks = []
    print(f"  [INFO] Splitting text...")

    for item in data:
        text = item.get("text", "")
        if not text: 
            continue
            
        source = item.get("url", "")
        title = item.get("title", "")
        category = item.get("category", "General")
        
        # Sanitize class name
        class_name = "".join(x for x in category.title() if x.isalnum())
        if not class_name: class_name = "General"

        chunks = splitter.split_text(text)
        for chunk in chunks:
            all_chunks.append({
                "class": class_name,
                "properties": {
                    "text": chunk,
                    "source": source,
                    "title": title,
                    "category": category
                }
            })

    print(f"  [INFO] Total chunks to ingest: {len(all_chunks)}")
    
    batch_size = 50
    print(f"  [INFO] Starting batch ingestion (Batch size: {batch_size})...")

    success_count = 0
    fail_count = 0

    for i in tqdm(range(0, len(all_chunks), batch_size), desc="  [INGEST]"):
        batch = all_chunks[i:i+batch_size]
        texts = [x["properties"]["text"] for x in batch]
        
        try:
            vectors = embedder.embed_documents(texts)
        except Exception as e:
            print(f"    [ERROR] Embedding failed: {e}")
            fail_count += len(batch)
            continue

        weaviate_objs = []
        for obj, vector in zip(batch, vectors):
            weaviate_objs.append({
                "class": obj["class"],
                "properties": obj["properties"],
                "vector": vector
            })
        
        try:
            r = requests.post(batch_endpoint, json={"objects": weaviate_objs})
            if r.status_code == 200:
                # Check for per-object errors in response
                results = r.json()
                for res in results:
                    if 'result' in res and 'errors' in res['result']:
                        fail_count += 1
                    else:
                        success_count += 1
            else:
                print(f"    [WARN] Batch request failed: {r.status_code} {r.text}")
                fail_count += len(batch)
        except Exception as e:
            print(f"    [ERROR] Batch connection failed: {e}")
            fail_count += len(batch)

    print(f"\n{'='*70}")
    print(f"  INGESTION COMPLETE")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    ingest()
