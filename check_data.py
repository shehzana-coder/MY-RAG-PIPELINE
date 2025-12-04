import weaviate
import json

# Connect to Weaviate using v4 syntax
client = weaviate.WeaviateClient(url="http://localhost:8081")

# 1. Check the schema
print("=== Weaviate Schema ===")
schema = client.schema.get()
print(json.dumps(schema, indent=4))  # Pretty print

# 2. Check stored data for each class
if "classes" in schema and schema["classes"]:
    print("\n=== Stored Data ===")
    for cls in schema["classes"]:
        class_name = cls["class"]
        print(f"\nClass: {class_name}")

        # Get all objects of this class
        objects = client.data_object.get(class_name=class_name)
        if objects.get("objects"):
            for obj in objects["objects"]:
                print(json.dumps(obj, indent=4))
        else:
            print(f"No objects stored in class '{class_name}'")
else:
    print("No classes found in the schema. You need to create a schema first.")
