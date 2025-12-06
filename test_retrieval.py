import asyncio
from query import retrieve_for_query_async

async def main():
    query = "tell me about dr ali shahid"
    print(f"Testing retrieval for: '{query}'")
    
    try:
        results = await retrieve_for_query_async(query)
        print(f"Found {len(results)} results.")
        for i, res in enumerate(results):
            print(f"[{i+1}] Score: {res.get('score')} | Source: {res.get('source')}")
            print(f"    Text: {res.get('text')[:100]}...")
    except Exception as e:
        print(f"Retrieval failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
