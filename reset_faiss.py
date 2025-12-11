# reset_faiss.py

import os
import asyncio
from query import ConnectionManager  # Make sure this points to your ConnectionManager class
from query import DistributedCache      # Your Redis cache class

async def reset_faiss_and_cache():
    """Clear FAISS index (memory + disk) and optionally clear cached embeddings in Redis."""
    
    # ------------------------------
    # 1. Clear FAISS in memory
    # ------------------------------
    faiss_index = await ConnectionManager.get_faiss_index()
    faiss_index.reset()
    print("[FAISS] Index cleared in memory.")

    # ------------------------------
    # 2. Remove FAISS index file from disk
    # ------------------------------
    faiss_index_path = "./cache/faiss_index.bin"
    if os.path.exists(faiss_index_path):
        os.remove(faiss_index_path)
        print(f"[FAISS] Index file '{faiss_index_path}' removed from disk.")
    else:
        print(f"[FAISS] No index file found at '{faiss_index_path}'.")

    # ------------------------------
    # 3. Clear cached embeddings from Redis
    # ------------------------------
    await DistributedCache.clear_pattern("emb:*")
    print("[Redis] Cached embeddings cleared.")

    print("[RESET COMPLETE] FAISS index and Redis cache reset successfully.")


if __name__ == "__main__":
    asyncio.run(reset_faiss_and_cache())
