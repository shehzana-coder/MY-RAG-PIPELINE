"""Create Weaviate schema classes per category file.

Usage:
    python weaviate_schema.py           # create classes for files in weaviate_ready_data/
    python weaviate_schema.py --drop    # drop then recreate classes

This script creates one Weaviate class per JSON file in `weaviate_ready_data/`.
Each class will have `vectorizer: 'none'` because the ingestion script
uploads vectors from the client side.
"""
import os
import re
import argparse
from dotenv import load_dotenv
import requests
from dotenv import load_dotenv

load_dotenv()

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
DATA_DIR = os.path.join(os.path.dirname(__file__), "weaviate_ready_data")


def slug_to_classname(slug: str) -> str:
    """Turn a filename/slug into a valid Weaviate ClassName.
    Rules: start with uppercase letter, only A-Z0-9 allowed.
    """
    name = os.path.splitext(os.path.basename(slug))[0]
    # remove non-alphanumeric
    parts = re.split(r"[^0-9a-zA-Z]+", name)
    parts = [p for p in parts if p]
    if not parts:
        return "WebPage"
    # CamelCase parts and ensure first char is alpha
    cname = "".join(p.capitalize() for p in parts)
    if not re.match(r"^[A-Za-z]", cname):
        cname = "C" + cname
    return cname


def class_exists(class_name: str) -> bool:
    """Check if a class exists via Weaviate REST API."""
    try:
        r = requests.get(f"{WEAVIATE_URL}/v1/schema")
        r.raise_for_status()
        schema = r.json()
        classes = schema.get("classes", [])
        exists = any(c.get("class") == class_name for c in classes)
        status = "✓ exists" if exists else "✗ does not exist"
        print(f"  [CHECK] Class '{class_name}' {status}")
        return exists
    except Exception as e:
        print(f"  [ERROR] Failed to check class '{class_name}': {e}")
        return False


def create_class(class_name: str) -> None:
    """Create a class using Weaviate REST API compatible with v4 server.
    Uses vectorizer 'none' so vectors are provided by client during ingestion.
    """
    schema = {
        "class": class_name,
        "vectorizer": "none",
        "properties": [
            {"name": "title", "dataType": ["text"]},
            {"name": "url", "dataType": ["text"]},
            {"name": "category", "dataType": ["text"]},
            {"name": "text", "dataType": ["text"]},
            {"name": "crawl_timestamp", "dataType": ["int"]},
        ],
    }
    try:
        print(f"  [CREATE] Creating class '{class_name}' with 5 properties...")
        r = requests.post(f"{WEAVIATE_URL}/v1/schema", json=schema)
        r.raise_for_status()
        print(f"  [✓ SUCCESS] Class '{class_name}' created successfully")
    except Exception as e:
        print(f"  [✗ FAILED] Could not create class '{class_name}': {e}")
        raise


def drop_class(class_name: str) -> None:
    try:
        print(f"  [DROP] Dropping class '{class_name}'...")
        r = requests.delete(f"{WEAVIATE_URL}/v1/schema/{class_name}")
        if r.status_code not in (200, 204, 404):
            r.raise_for_status()
        print(f"  [✓ SUCCESS] Class '{class_name}' dropped successfully")
    except Exception as e:
        print(f"  [✗ FAILED] Could not drop class '{class_name}': {e}")


def main(drop_existing: bool = False):
    print(f"\n{'='*60}")
    print(f"Weaviate Schema Creator")
    print(f"{'='*60}")
    print(f"Weaviate URL: {WEAVIATE_URL}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Drop existing: {drop_existing}")
    print(f"{'='*60}\n")
    
    if not os.path.isdir(DATA_DIR):
        print(f"[ERROR] No directory found: {DATA_DIR}")
        return

    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.json')]
    if not files:
        print(f"[INFO] No JSON files in {DATA_DIR}, using fallback...")
        files = ["webpage.json"]
    
    print(f"[INFO] Found {len(files)} JSON file(s) to process:")
    for f in files:
        print(f"  - {f}")
    print()

    created = []
    for idx, f in enumerate(files, 1):
        class_name = slug_to_classname(f)
        print(f"[{idx}/{len(files)}] Processing: {f}")
        
        if class_exists(class_name):
            if drop_existing:
                drop_class(class_name)
                print(f"  [RETRY] Creating class '{class_name}'...")
                create_class(class_name)
                created.append(class_name)
            else:
                print(f"  [SKIP] Class already exists, skipping")
        else:
            create_class(class_name)
            created.append(class_name)
        print()

    print(f"{'='*60}")
    if created:
        print(f"✓ SUMMARY: Successfully created {len(created)} class(es)")
        for cls in created:
            print(f"  ✓ {cls}")
    else:
        print(f"✗ SUMMARY: No classes created.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Weaviate schema classes per category file.")
    parser.add_argument("--drop", action="store_true", help="Drop existing classes before creating")
    args = parser.parse_args()
    main(drop_existing=args.drop)
