import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain.agents import create_agent

from query import perform_query

load_dotenv()

import time
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

class TimingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.start_time = None

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.start_time = time.time()
        print("  [TIMING] Agent Thinking (LLM Start)...")

    def on_llm_end(self, response: LLMResult, **kwargs):
        end_time = time.time()
        if self.start_time:
            elapsed = end_time - self.start_time
            print(f"  [TIMING] Agent Thinking Finished in {elapsed:.4f}s")



def main():
    print(f"\n{'='*70}")
    print(f"  RAG Agent — Weaviate + LangGraph")
    print(f"{'='*70}\n")

    # Initialize LLM with streaming callback
    try:
        from langchain_community.callbacks import StreamingStdOutCallbackHandler
    except ImportError:
        from langchain_core.callbacks import StreamingStdOutCallbackHandler

    llm = ChatOpenAI(temperature=0, model=os.environ.get('LLM_MODEL', 'gpt-3.5-turbo'), streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
    print("  [INFO] ChatOpenAI initialized\n")

    # Wrap perform_query as a Tool
    search_tool = Tool(
        name="weaviate_query",
        func=lambda q: perform_query(q),
        description=(
            "Search the ingested website content stored in Weaviate. "
            "Input: a search query. Output: relevant text chunks."
        ),
        return_direct=True
    )
    tools = [search_tool]

    # Create a system prompt template (replaces state_modifier)
    system_prompt = "You are a helpful assistant. If you use the weaviate_query tool, its output IS the final answer. Do not add any additional text."

    # Create the agent using the new LangChain method
    agent_executor = create_agent(
    tools=tools,
    system_prompt=system_prompt,
    model=llm,
)


    print("  Agent ready. Type a question (or 'exit' to quit):\n")
    while True:
        try:
            q = input("  > ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit"):
                print("\n  Exiting agent...\n")
                break

            print(f"\n  [AGENT] Processing: '{q}'")
            
            # Since we have streaming enabled:
            # 1. If agent decides to talk directly (no tool), it streams to stdout via Callback.
            # 2. If agent uses tool, the tool (perform_query) streams to stdout manually.
            # We assume the agent output is already printed.
            
            response = agent_executor.invoke({"messages": [("user", q)]})
            
            # We don't print the final message here to avoid duplication.
            # But just in case nothing was printed (e.g. silent failure?), we could check.
            # Ideally, streaming handles it.
            print("\n") # Ensure newline after streaming


        except KeyboardInterrupt:
            print("\n\n  Interrupted by user.\n")
            break
        except Exception as e:
            print(f"\n  [ERROR] {e}\n")



if __name__ == "__main__":
    main()
