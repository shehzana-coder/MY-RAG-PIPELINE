import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

from query import perform_query

load_dotenv()


def main():
    print(f"\n{'='*70}")
    print(f"  RAG Agent — Weaviate + LangChain")
    print(f"{'='*70}\n")

    # Initialize LLM
    llm = ChatOpenAI(temperature=0, model=os.environ.get('LLM_MODEL', 'gpt-3.5-turbo'))
    print("  [INFO] ChatOpenAI initialized\n")

    # Wrap perform_query as a Tool the agent can call
    search_tool = Tool(
        name="weaviate_query",
        func=lambda q: perform_query(q, k=5, filters=None),
        description=(
            "Search the ingested website content stored in Weaviate with metadata filters. "
            "Input: a search query. Output: relevant text chunks with source and category."
        ),
    )

    tools = [search_tool]

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=5,
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
            resp = agent.run(q)
            print(f"\n  [RESPONSE]\n  {resp}\n")
        except KeyboardInterrupt:
            print("\n\n  Interrupted by user.\n")
            break
        except Exception as e:
            print(f"\n  [ERROR] {e}\n")


if __name__ == "__main__":
    main()
