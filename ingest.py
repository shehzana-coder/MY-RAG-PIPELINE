import os
import glob
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import requests

from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

load_dotenv()

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
EMBED_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "openai")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
JSON_DATA_DIR = os.path.join(os.path.dirname(__file__), "weaviate_ready_data")
os.makedirs(DATA_DIR, exist_ok=True)


def slug_to_classname(slug: str) -> str:
    """Convert filename/slug to Weaviate ClassName (matches weaviate_schema.py)."""
    name = os.path.splitext(os.path.basename(slug))[0]
    parts = re.split(r"[^0-9a-zA-Z]+", name)
    parts = [p for p in parts if p]
    if not parts:
        return "WebPage"
    cname = "".join(p.capitalize() for p in parts)
    if not re.match(r"^[A-Za-z]", cname):
        cname = "C" + cname
    return cname


def get_category_from_filename(filename: str) -> str:
    """Extract category slug from data filename (e.g., 'category__page__hash.txt' -> 'category')."""
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("__")
    return parts[0] if parts else "not-categorized"


def ensure_schema():
    print("\n[INFO] Schema creation is now handled by weaviate_schema.py")
    print("[INFO] Skipping schema setup — assuming classes already exist in Weaviate.\n")


def load_file_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def get_embeddings_provider():
    if EMBED_PROVIDER == "openai":
        return OpenAIEmbeddings()
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        model = os.environ.get("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name=model)


def upsert_object_to_weaviate(class_name: str, properties: dict, vector: list) -> None:
    """Upsert an object to Weaviate using REST API (v4 compatible)."""
    obj = {
        "class": class_name,
        "properties": properties,
        "vector": vector,
    }
    try:
        r = requests.post(
            f"{WEAVIATE_URL}/v1/objects",
            json=obj,
            timeout=10
        )
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"[WARN] Failed to upsert to {class_name}: {e}")
        return False


def ingest():
    print(f"\n{'='*70}")
    print(f"  Weaviate Ingestion — Per-Category Classes")
    print(f"{'='*70}")
    print(f"  Weaviate URL: {WEAVIATE_URL}")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Embedding provider: {EMBED_PROVIDER}")
    print(f"{'='*70}\n")
    
    ensure_schema()
    
    files = sorted(glob.glob(os.path.join(DATA_DIR, "**/*.txt"), recursive=True))
    if not files:
        print(f"  [ERROR] No .txt files found in {DATA_DIR}. Run scraper first.")
        return

    print(f"  [INFO] Found {len(files)} .txt file(s) to ingest\n")

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    print(f"  [INFO] Initializing text splitter (chunk_size=1000, overlap=200)")
    
    embedder = get_embeddings_provider()
    print(f"  [INFO] Initializing embeddings provider: {EMBED_PROVIDER}\n")

    batch_size = 32
    total_ingested = 0
    total_failed = 0
    file_stats = []

    for file_idx, path in enumerate(files, 1):
        print(f"  [{'='*60}]")
        print(f"  [{file_idx}/{len(files)}] Processing: {os.path.basename(path)}")
        
        text = load_file_text(path)
        print(f"    [LOAD] Read file ({len(text)} characters)")
        
        category_slug = get_category_from_filename(path)
        class_name = slug_to_classname(category_slug)
        print(f"    [MAP] Category: {category_slug} → Class: {class_name}")
        
        chunks = splitter.split_text(text)
        print(f"    [SPLIT] Created {len(chunks)} text chunks")
        source = os.path.basename(path)

        file_ingested = 0
        file_failed = 0

        for i in tqdm(range(0, len(chunks), batch_size), desc=f"    [EMBED]", leave=False):
            batch_chunks = chunks[i:i+batch_size]
            try:
                print(f"      → Embedding batch {i//batch_size + 1} ({len(batch_chunks)} chunks)...")
                vectors = embedder.embed_documents(batch_chunks)
                print(f"        ✓ Got {len(vectors)} embedding vectors")
                
                for chunk_idx, (chunk_text, vector) in enumerate(zip(batch_chunks, vectors)):
                    props = {
                        "text": chunk_text,
                        "source": source,
                        "category": category_slug,
                    }
                    success = upsert_object_to_weaviate(class_name, props, vector)
                    if success:
                        file_ingested += 1
                        total_ingested += 1
                    else:
                        file_failed += 1
                        total_failed += 1
            except Exception as e:
                print(f"      [ERROR] Failed to embed batch: {e}")
                file_failed += len(batch_chunks)
                total_failed += len(batch_chunks)
                break

        file_stats.append({
            'file': os.path.basename(path),
            'category': category_slug,
            'ingested': file_ingested,
            'failed': file_failed
        })
        print(f"    [DONE] Ingested {file_ingested} objects (Failed: {file_failed})\n")

    print(f"\n{'='*70}")
    print(f"  INGESTION SUMMARY")
    print(f"{'='*70}")
    print(f"  Total files processed: {len(files)}")
    print(f"  Total objects ingested: {total_ingested}")
    print(f"  Total objects failed: {total_failed}")
    
    if file_stats:
        print(f"\n  Per-file breakdown:")
        for stat in file_stats[:10]:  # Show first 10 files
            status = f"✓ {stat['ingested']}" if stat['ingested'] > 0 else f"✗ Failed"
            print(f"    {stat['file']:50} → {status}")
        if len(file_stats) > 10:
            print(f"    ... and {len(file_stats) - 10} more files")
    
    print(f"\n{'='*70}")
    print(f"  ✓ Ingestion complete!\n")


if __name__ == "__main__":
    ingest()
