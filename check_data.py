import weaviate
import json

def check_weaviate_data():
    # Connect to Weaviate using v4 syntax
    client = weaviate.connect_to_local(port=8081)

    try:
        # 1. Check the schema (Collections in v4)
        print("=== Weaviate Collections (Schema) ===")
        collections = client.collections.list_all()
        
        if not collections:
            print("No collections (classes) found in the schema.")
            return

        for name in collections:
            print(f"- {name}")

        # 2. Check stored data for each collection
        print("\n=== Stored Data ===")
        for name in collections:
            print(f"\nCollection: {name}")
            collection = client.collections.get(name)
            
            # Fetch objects (limit to 10 for readability)
            response = collection.query.fetch_objects(limit=10)
            
            if response.objects:
                print(f"Found {len(response.objects)} objects (showing first 10):")
                for i, obj in enumerate(response.objects):
                    print(f"\n--- Object {i+1} ---")
                    print(json.dumps(obj.properties, indent=4))
            else:
                print(f"No objects stored in collection '{name}'")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    check_weaviate_data()
