import asyncio
import redis.asyncio as redis

async def clear_cache():
    r = await redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    await r.flushdb()
    await r.close()
    print("Cache cleared!")

asyncio.run(clear_cache())
