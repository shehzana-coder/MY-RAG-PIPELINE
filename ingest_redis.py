import os
import json
import glob
import time
import redis
from redis.commands.search.field import TextField, VectorField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Configuration
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
DATA_DIR = os.path.join(os.getcwd(), "weaviate_ready_data")
VECTOR_DIM = 1536  # OpenAI embeddings dimension
INDEX_NAME = "namal_idx"

def get_redis_client():
    return redis.from_url(REDIS_URL, decode_responses=True)

def create_index(r):
    try:
        r.ft(INDEX_NAME).info()
        print(f"Index '{INDEX_NAME}' already exists.")
    except:
        print(f"Creating index '{INDEX_NAME}'...")
        schema = (
            TextField("text"),
            TextField("source"),
            TextField("category"),
            TextField("title"),
            VectorField("vector",
                "FLAT", {
                    "TYPE": "FLOAT32",
                    "DIM": VECTOR_DIM,
                    "DISTANCE_METRIC": "COSINE",
                }
            ),
        )
        definition = IndexDefinition(prefix=["doc:"], index_type=IndexType.HASH)
        r.ft(INDEX_NAME).create_index(schema, definition=definition)
        print("Index created successfully.")

def ingest_data():
    r = get_redis_client()
    
    # 1. Create Index
    create_index(r)

    # 2. Load Embedder
    print("Initializing Embedder...")
    embedder = OpenAIEmbeddings()

    # 3. Read Files
    files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    if not files:
        print(f"No JSON files found in {DATA_DIR}")
        return

    print(f"Found {len(files)} files to ingest.")
    
    total_docs = 0
    pipe = r.pipeline()
    
    for file_path in files:
        class_name = os.path.basename(file_path).replace(".json", "")
        print(f"Processing {class_name}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for i, item in enumerate(data):
            text = item.get("text", "")
            if not text:
                continue

            # Generate Embedding
            # Note: For production, batch embedding is better. Doing one-by-one here for simplicity/code reuse.
            try:
                vector = embedder.embed_query(text)
                vector_bytes = np.array(vector, dtype=np.float32).tobytes()
            except Exception as e:
                print(f"Failed to embed document {i} in {class_name}: {e}")
                continue

            # Redis Key
            key = f"doc:{class_name}:{i}"
            
            # Metadata
            mapping = {
                "text": text,
                "source": item.get("source", ""),
                "title": item.get("title", ""),
                "category": class_name, # Storing class name as category
                "vector": vector_bytes
            }
            
            pipe.hset(key, mapping=mapping)
            total_docs += 1
            
            if total_docs % 100 == 0:
                print(f"  Indexed {total_docs} documents...")
                pipe.execute()

    pipe.execute() # Flush remaining
    print(f"\nIngestion Complete. Total Documents: {total_docs}")

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found. Please run scraper/ingest first.")
    else:
        ingest_data()
