import weaviate
import os
import json
from dotenv import load_dotenv

load_dotenv()

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://127.0.0.1:8081")

def main():
    print(f"Connecting to {WEAVIATE_URL}...")
    client = weaviate.Client(WEAVIATE_URL)
    
    # 1. List all classes
    schema = client.schema.get()
    classes = [c['class'] for c in schema.get('classes', [])]
    print(f"Found {len(classes)} classes: {classes}")
    
    # 2. Search for 'Ali Shahid' in each class
    query_text = "Ali Shahid"
    found_any = False
    
    print(f"\nSearching for '{query_text}'...")
    
    for cls in classes:
        try:
            # Simple nearText search
            response = (
                client.query
                .get(cls, ["text", "source", "title"])
                .with_hybrid(query=query_text)
                .with_limit(3)
                .do()
            )
            
            get_resp = response.get('data', {}).get('Get', {}).get(cls, [])
            if get_resp:
                print(f"\n[Class: {cls}] Found {len(get_resp)} hits:")
                for hit in get_resp:
                    print(f" - {hit}")
                found_any = True
        except Exception as e:
            print(f"[Error] querying {cls}: {e}")

    if not found_any:
        print("\n[RESULT] No documents found for 'Ali Shahid' in any class.")
    else:
        print("\n[RESULT] Found documents above.")

if __name__ == "__main__":
    main()
