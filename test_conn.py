import httpx
import asyncio
import time

async def test_url(url):
    print(f"Testing {url} ...")
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{url}/v1/meta")
            print(f"[{resp.status_code}] {url} (took {time.time() - start:.4f}s)")
            print(resp.json())
    except Exception as e:
        print(f"[FAIL] {url}: {e} (took {time.time() - start:.4f}s)")

async def main():
    await test_url("http://localhost:8081")
    await test_url("http://127.0.0.1:8081")

if __name__ == "__main__":
    asyncio.run(main())
