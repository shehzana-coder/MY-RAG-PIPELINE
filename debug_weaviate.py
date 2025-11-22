import requests
import os
from dotenv import load_dotenv

load_dotenv()

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8081")

print(f"Checking Weaviate at: {WEAVIATE_URL}")

try:
    r = requests.get(f"{WEAVIATE_URL}/v1/schema", timeout=5)
    print(f"Status Code: {r.status_code}")
    if r.status_code == 200:
        schema = r.json()
        classes = [c.get("class") for c in schema.get("classes", [])]
        print(f"Available Classes: {classes}")
        
        # Check object count for each class
        for c in classes:
            gql = f"{{ Aggregate {{ {c} {{ meta {{ count }} }} }} }}"
            r_count = requests.post(f"{WEAVIATE_URL}/v1/graphql", json={"query": gql})
            print(f"Class '{c}' count: {r_count.json()}")
            
    else:
        print(f"Error response: {r.text}")
except Exception as e:
    print(f"Connection failed: {e}")
