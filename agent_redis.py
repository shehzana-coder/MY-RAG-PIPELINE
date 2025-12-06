import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain.agents import create_agent
from query_redis import perform_query
import time
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

load_dotenv()

class TimingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.start_time = None
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.start_time = time.time()
        print("  [TIMING] Agent Thinking (LLM Start)...")
    def on_llm_end(self, response: LLMResult, **kwargs):
        if self.start_time:
            print(f"  [TIMING] Agent Thinking Finished in {time.time() - self.start_time:.4f}s")

def main():
    print(f"\n{'='*70}")
    print(f"  Redis RAG Agent")
    print(f"{'='*70}\n")

    # Initialize LLM
    try:
        from langchain_community.callbacks import StreamingStdOutCallbackHandler
    except ImportError:
        from langchain_core.callbacks import StreamingStdOutCallbackHandler

    llm = ChatOpenAI(temperature=0, model=os.environ.get('LLM_MODEL', 'gpt-3.5-turbo'), streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
    print("  [INFO] ChatOpenAI initialized\n")

    # Wrap perform_query as a Tool
    search_tool = Tool(
        name="redis_query",
        func=lambda q: perform_query(q),
        description="Search stored content in Redis. Input: query. Output: answer.",
        return_direct=True
    )
    tools = [search_tool]

    system_prompt = "You are a helpful assistant. If you use the redis_query tool, its output IS the final answer."

    agent_executor = create_agent(tools=tools, system_prompt=system_prompt, model=llm)

    print("  Agent ready. Type a question (or 'exit' to quit):\n")
    while True:
        try:
            q = input("  > ").strip()
            if not q: continue
            if q.lower() in ("exit", "quit"): break

            print(f"\n  [AGENT] Processing: '{q}'")
            agent_executor.invoke({"messages": [("user", q)]})
            print("\n")

        except KeyboardInterrupt:
            print("\n\n  Interrupted by user.\n")
            break
        except Exception as e:
            print(f"\n  [ERROR] {e}\n")

if __name__ == "__main__":
    main()
