from seleniumbase import Driver
from urllib.parse import urljoin, urlparse, urldefrag
import json
import time
from bs4 import BeautifulSoup

def remove_duplicate_links_across_categories(input_file):
    """
    Reads a JSON file with category->links structure,
    ensures each link appears in only one category.
    Overwrites the input file with cleaned data.
    """
    # Load JSON data
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Dictionary to track where each link should finally appear
    link_to_category = {}

    # First pass: map each link to its latest category
    for category, links in data.items():
        for link in links:
            link_to_category[link] = category

    # Second pass: rebuild categories using the latest category mapping
    cleaned_data = {}
    for category, links in data.items():
        cleaned_links = []
        for link in links:
            if link_to_category[link] == category:
                cleaned_links.append(link)
        if cleaned_links:
            cleaned_data[category] = cleaned_links

    # Overwrite the input file with cleaned data
    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=4, ensure_ascii=False)

    print(f"✅ Cleaned JSON saved and overwritten in: {input_file}")
    return cleaned_data


def normalize_url(url):
    """Normalize URLs: remove fragments and trailing slashes."""
    full_url, _ = urldefrag(url)
    if full_url.endswith("/"):
        full_url = full_url.rstrip("/")
    return full_url


def crawl_navbar_links_with_seleniumbase(base_url, delay=2.0, output_file="navbar_links.json"):
    """
    Crawl navbar links using SeleniumBase driver in undetected mode.
    Extracts navbar categories, deduplicates links, and categorizes body links.
    """
    print("🚀 Starting navbar crawl with SeleniumBase (undetected mode)...", flush=True)
    start_time = time.time()

    navbar_data = {}
    all_links_set = set()

    driver = Driver(browser="chrome", headless=True, uc=True)
    try:
        driver.open(base_url)
        driver.sleep(delay)

        html_content = driver.get_page_source()
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract navbar links
        navbars = soup.find_all("nav")
        for nav in navbars:
            items = nav.find_all("li")
            for item in items:
                a_tag = item.find("a")
                if not a_tag:
                    continue
                
                heading_text = a_tag.get_text(strip=True)
                if not heading_text:
                    continue

                if heading_text not in navbar_data:
                    navbar_data[heading_text] = set()

                # Get all links inside this menu item
                all_a_tags = item.find_all("a")
                for a in all_a_tags:
                    href = a.get("href")
                    if not href:
                        continue

                    full_link = urljoin(base_url, href.strip())
                    full_link = normalize_url(full_link)

                    # Skip unwanted file types
                    if any(full_link.lower().endswith(ext) for ext in 
                           [".jpg", ".jpeg", ".png", ".gif", ".pdf", ".mp4", ".webp"]):
                        continue

                    navbar_data[heading_text].add(full_link)
                    all_links_set.add(full_link)

        # Extract all body links (not in navbar/footer)
        body = soup.find("body")
        if body:
            # Remove navbar and footer from body
            for nav in body.find_all("nav"):
                nav.decompose()
            for footer in body.find_all("footer"):
                footer.decompose()

            all_body_links = set()
            for a in body.find_all("a"):
                href = a.get("href")
                if not href:
                    continue

                full_link = urljoin(base_url, href.strip())
                full_link = normalize_url(full_link)

                # Skip unwanted file types
                if any(full_link.lower().endswith(ext) for ext in 
                       [".jpg", ".jpeg", ".png",".webp", ".gif", ".pdf", ".mp4"]):
                    continue

                all_body_links.add(full_link)

            # Links not in navbar categories
            not_categorized_links = all_body_links - all_links_set
            if not_categorized_links:
                navbar_data["Not Categorized"] = not_categorized_links

    finally:
        driver.quit()

    end_time = time.time()
    print(f"\nNavbar crawl completed in {end_time - start_time:.2f} seconds.", flush=True)

    # Remove duplicates within categories
    for key in navbar_data:
        navbar_data[key] = set(navbar_data[key])

    clean_data = {k: sorted(list(v)) for k, v in navbar_data.items() if v}
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, indent=4, ensure_ascii=False)

    print(f"Found {len(clean_data)} navbar categories (including 'Not Categorized' if present).")
    print(f"Links saved to: {output_file}")

    # Remove cross-category duplicates
    clean_data = remove_duplicate_links_across_categories(output_file)

    return clean_data


def crawl_sub_links_with_seleniumbase(input_file, base_url, delay=1.0, output_file=None):
    """
    Crawl sub-links using SeleniumBase driver in undetected mode.
    Structure: {category: {link: [link, sub_link1, sub_link2, ...]}}
    """
    if output_file is None:
        output_file = input_file

    with open(input_file, "r", encoding="utf-8") as f:
        navbar_data = json.load(f)

    domain = urlparse(base_url).netloc

    # Track all previously seen links to avoid duplicates
    existing_links = set()
    for links in navbar_data.values():
        for l in links:
            full, _ = urldefrag(l.strip())
            existing_links.add(full)

    expanded_data = {}
    driver = Driver(browser="chrome", headless=True, uc=True)

    try:
        for category, links in navbar_data.items():
            expanded_data[category] = {}

            for link in links:
                link, _ = urldefrag(link.strip())

                # Skip non-domain links
                if urlparse(link).netloc != domain:
                    continue

                try:
                    print(f"🔄 Crawling: {category} → {link}")
                    driver.open(link)
                    driver.sleep(delay)

                    html_content = driver.get_page_source()
                    soup = BeautifulSoup(html_content, "html.parser")

                    # Remove navbar and footer from body to avoid duplicate links
                    body = soup.find("body")
                    if body:
                        for nav in body.find_all("nav"):
                            nav.decompose()
                        for footer in body.find_all("footer"):
                            footer.decompose()

                    sub_links = set()

                    # Extract all links from body
                    if body:
                        for a in body.find_all("a"):
                            href = a.get("href")
                            if not href:
                                continue

                            full_link = urljoin(base_url, href.strip())
                            full_link, _ = urldefrag(full_link)

                            # Skip external links, duplicates, images/docs, etc.
                            if (full_link in existing_links or
                                urlparse(full_link).netloc != domain or
                                any(full_link.lower().endswith(ext) for ext in
                                    [".jpg", ".jpeg", ".png", ".gif", ".pdf", ".mp4", ".zip"])):
                                continue

                            sub_links.add(full_link)
                            existing_links.add(full_link)

                    expanded_data[category][link] = sorted([link] + list(sub_links))
                    print(f"   ✅ Found {len(sub_links)} sub-links")

                except Exception as e:
                    print(f"   ❌ Failed to crawl {link}: {e}")
                    expanded_data[category][link] = [link]

    finally:
        driver.quit()

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(expanded_data, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Sub-links saved to: {output_file}")
    return expanded_data



# ----------------------------
# Main
# ----------------------------
def main():
    base_url = "https://namal.edu.pk"
    start_time = time.time()

    print("=" * 60)
    print("🌐 Web Crawler with SeleniumBase (Undetected Mode)")
    print("=" * 60)

    # Step 1: Crawl navbar links
    crawl_navbar_links_with_seleniumbase(base_url, output_file="navbar_links.json")

    # Step 2: Crawl sub-links
    crawl_sub_links_with_seleniumbase("navbar_links.json", base_url, output_file="navbar_links.json")

    total_time = time.time() - start_time
    print(f"\n⏱️ Total crawling time: {total_time:.2f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    main()




