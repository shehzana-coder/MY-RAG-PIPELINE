import json
import os
import re
import time
import uuid
import hashlib
from pathlib import Path
from bs4 import BeautifulSoup
from seleniumbase import Driver
from urllib.parse import urlparse, urljoin, urldefrag


def slugify(text):
    """Convert text into safe filename format."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return text.strip("-")


def extract_heading_blocks(soup):
    """
    Extract sections based on headings (H1–H6) with their associated content.
    Output: List of {heading, content_html, h_level}
    """

    body = soup.find("body")
    if not body:
        return []

    headings = body.find_all(re.compile("^h[1-6]$"))
    results = []

    for i, h in enumerate(headings):
        heading_text = h.get_text(strip=True)
        h_level = int(h.name[1])  # h2 -> 2

        content_parts = []
        node = h.find_next_sibling()

        while node and node.name not in ["h1","h2","h3","h4","h5","h6"]:
            content_parts.append(str(node))
            node = node.find_next_sibling()

        results.append({
            "heading": heading_text,
            "content_html": "\n".join(content_parts).strip(),
            "h_level": h_level
        })

    return results


def chunk_text(text, chunk_size=1200):
    """Splits text into clean, semantic chunks."""
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end

    return chunks


def scrape_for_weaviate(
        sub_links_json,
        base_url,
        delay=2.0,
        output_dir="data",
        organization_id=None,
    ):
    """
    Scrapes subpage content and produces PERFECT JSON chunks
    ready for Weaviate ingestion.
    Now includes:
    ✔ Full <body> content as first block (with page top heading)
    """

    # Load JSON if file path provided
    if isinstance(sub_links_json, str):
        with open(sub_links_json, "r", encoding="utf-8") as f:
            sub_links_json = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    # Also create the `data/` folder used by ingest.py
    data_dir = Path(os.path.dirname(__file__)) / "data"
    data_dir.mkdir(exist_ok=True)

    driver = Driver(browser="chrome", headless=True, uc=True)

    final_output = {}

    # Deduplication trackers
    seen_urls = set()
    seen_hashes = set()

    def extract_section_for_category(soup, category_name):
        """Try to find sections whose heading matches the category name.
        If found, return the concatenated text under those headings. Otherwise return None.
        """
        body = soup.find("body")
        if not body:
            return None

        # Remove header/nav/footer to avoid repeated site chrome
        for node in body.find_all(["nav", "header", "footer", "aside"]):
            node.decompose()

        # Search headings for matches to category name
        matches = []
        for level in range(1, 7):
            for h in body.find_all(f"h{level}"):
                ht = h.get_text(" ", strip=True)
                if not ht:
                    continue
                # match if heading contains category text (case-insensitive)
                if category_name.strip().lower() in ht.strip().lower():
                    # collect nodes until next heading of same-or-higher level
                    parts = []
                    node = h.find_next_sibling()
                    while node and (not (hasattr(node, 'name') and re.match(r"^h[1-6]$", node.name))):
                        parts.append(str(node))
                        node = node.find_next_sibling()
                    text = BeautifulSoup("\n".join(parts), "html.parser").get_text(" ", strip=True)
                    if text:
                        matches.append(text)

        if matches:
            return "\n\n".join(matches)

        # Fallback: look for <main> or <article>
        main_tag = body.find("main") or body.find("article")
        if main_tag:
            return main_tag.get_text(" ", strip=True)

        # Last fallback: the body text (already header/footer removed)
        return body.get_text(" ", strip=True)

    for category, pages in sub_links_json.items():

        print(f"\n📂 Category: {category}")
        cat_slug = slugify(category)
        category_file = os.path.join(output_dir, f"{cat_slug}.json")

        category_records = []

        # pages is expected to be a mapping of page -> [link, sublinks...]
        for page_url, sub_links in pages.items():
            for link in sub_links:
                try:
                    # Normalize URL (remove fragments)
                    norm = urldefrag(urljoin(base_url, link))[0]
                    if norm in seen_urls:
                        print(f"  ↩️ Skipping duplicate URL: {norm}")
                        continue

                    print(f"  🟦 Scraping: {norm}")
                    driver.get(norm)
                    time.sleep(delay)

                    soup = BeautifulSoup(driver.page_source, "html.parser")

                    # Try to extract the section most relevant to this category
                    section_text = extract_section_for_category(soup, category)
                    if not section_text:
                        print(f"    ⚠️ No content found for {norm}, skipping")
                        seen_urls.add(norm)
                        continue

                    # Deduplicate by full-page section hash
                    h = hashlib.sha256(section_text.encode("utf-8")).hexdigest()
                    if h in seen_hashes:
                        print(f"    ↩️ Skipping duplicate content for {norm}")
                        seen_urls.add(norm)
                        continue
                    seen_hashes.add(h)

                    # Page title
                    page_title = soup.title.string.strip() if soup.title and soup.title.string else category

                    record = {
                        "id": str(uuid.uuid4()),
                        "title": page_title,
                        "url": norm,
                        "category": category,
                        "text": section_text,
                        "crawl_timestamp": int(time.time()),
                    }

                    category_records.append(record)

                    # Save a plain .txt for quick ingestion
                    parsed = urlparse(norm)
                    page_slug = slugify(parsed.path.strip('/')) or slugify(page_title) or parsed.netloc
                    txt_name = f"{cat_slug}__{page_slug}__{h[:8]}.txt"
                    with open(data_dir / txt_name, "w", encoding="utf-8") as of:
                        of.write(section_text)

                    # mark URL
                    seen_urls.add(norm)

                except Exception as e:
                    print(f"     ❌ Error scraping {link}: {e}")

        # Write category-level JSON for reference (clean + readable)
        with open(category_file, "w", encoding="utf-8") as f:
            json.dump(category_records, f, indent=2, ensure_ascii=False)

        final_output[category] = category_file
        print(f"  ✔ Saved category → {category_file}")

    driver.quit()
    print(f"\n🎉 Scraping completed. Wrote cleaned category JSON to: {output_dir}")

    return final_output

scrape_for_weaviate(
    sub_links_json="navbar_links.json",
    base_url="https://namal.edu.pk",
    delay=2,
    output_dir="weaviate_ready_data",
    organization_id="org-123"   # optional
)
