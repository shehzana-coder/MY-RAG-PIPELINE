import httpx
import asyncio
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Force 127.0.0.1 to avoid localhost issues
WEAVIATE_URL = "http://127.0.0.1:8081"

async def main():
    print(f"Connecting to {WEAVIATE_URL}...")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Check only DepartmentOfComputerScience
            cls = "DepartmentOfComputerScience"
            query_text = "Ali Shahid"
            print(f"\nSearching for '{query_text}' in {cls}...")

            gql = f"""
            {{
                Get {{
                    {cls}(
                        hybrid: {{
                            query: "{query_text}"
                        }}
                        limit: 3
                    ) {{
                        text
                        source
                        title
                        _additional {{ score }}
                    }}
                }}
            }}
            """
            
            r = await client.post(f"{WEAVIATE_URL}/v1/graphql", json={"query": gql})
            # print(r.text) # Debug raw response if needed
            r.raise_for_status()
            
            data = r.json().get("data", {}).get("Get", {}).get(cls, [])
            
            if data:
                print(f"[SUCCESS] Found {len(data)} hits for '{query_text}':")
                for item in data:
                    print(f" - Title: {item.get('title')}")
                    print(f"   Score: {item.get('_additional', {}).get('score')}")
                    print(f"   Excerpt: {item.get('text')[:100]}...")
            else:
                print(f"[FAIL] No hits for '{query_text}' in {cls}.")
                
                # Check if class is empty
                print("Checking if class has ANY data...")
                gql_dump = f"{{ Get {{ {cls}(limit: 1) {{ text }} }} }}"
                r2 = await client.post(f"{WEAVIATE_URL}/v1/graphql", json={"query": gql_dump})
                print(f"Dump result: {r2.json()}")

    except Exception as e:
        print(f"[FAIL] Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
