import os
import json
import time
import glob
from typing import List, Dict
from dotenv import load_dotenv
from tqdm import tqdm
import requests
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://127.0.0.1:8081")
EMBED_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "openai")
# Path to the directory containing JSON files
DATA_DIR = os.path.join(os.path.dirname(__file__), "weaviate_ready_data")

def get_embedder():
    if EMBED_PROVIDER == "openai":
        return OpenAIEmbeddings()
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        model = os.environ.get("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name=model)

def ingest():
    print(f"\n{'='*70}")
    print(f"  Weaviate Ingestion (Multi-file)")
    print(f"{'='*70}")
    print(f"  Target: {WEAVIATE_URL}")
    print(f"  Source Directory: {DATA_DIR}")

    if not os.path.exists(DATA_DIR):
        print(f"  [ERROR] Directory not found: {DATA_DIR}")
        return

    # Get all JSON files
    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    print(f"  [INFO] Found {len(json_files)} JSON files to process.")

    embedder = get_embedder()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    batch_endpoint = f"{WEAVIATE_URL}/v1/batch/objects"
    
    all_chunks = []
    
    # Process each file
    for file_path in tqdm(json_files, desc="  [READING FILES]"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Handle case where file might be empty or not a list
            if not isinstance(data, list):
                # If it's a dict, wrap it in a list
                if isinstance(data, dict):
                    data = [data]
                else:
                    continue

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
        except Exception as e:
            print(f"  [WARN] Failed to process {os.path.basename(file_path)}: {e}")

    print(f"  [INFO] Total chunks to ingest: {len(all_chunks)}")
    
    if not all_chunks:
        print("  [WARN] No data found to ingest.")
        return

    batch_size = 50
    print(f"  [INFO] Starting batch ingestion (Batch size: {batch_size})...")

    success_count = 0
    fail_count = 0

    for i in tqdm(range(0, len(all_chunks), batch_size), desc="  [INGESTING BATCHES]"):
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
