import os
from dotenv import load_dotenv
import re
import requests
import json
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
load_dotenv()  
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
EMBED_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "openai")
ORG_NAME = os.environ.get("ORG_NAME", "Your Organization")


def is_greeting(text: str) -> bool:
    text = text.strip().lower()
    # simple greeting detection
    return bool(re.match(r"^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening)|thanks|thank you)\b", text))


def get_embedder():
    if EMBED_PROVIDER == "openai":
        return OpenAIEmbeddings()
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        model = os.environ.get("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name=model)


def get_available_classes() -> List[str]:
    try:
        r = requests.get(f"{WEAVIATE_URL}/v1/schema", timeout=5)
        r.raise_for_status()
        schema = r.json()
        return [c.get("class") for c in schema.get("classes", [])]
    except Exception:
        return []


def analyze_query_intent(user_query: str) -> Dict:
    """Use the LLM to decide whether to (a) answer directly, (b) ask a clarifying question, or (c) expand into sub-queries.

    Returns a dict: {"action": "direct"|"clarify"|"expand", "clarification": str|None, "subqueries": [str]|None}
    """
    llm = ChatOpenAI(temperature=0)
    prompt = (
        "You are a helpful assistant that classifies user queries for retrieval.\n"
        "Given the user query, decide whether the query is: \n"
        "- 'direct' meaning it is clear and should be searched as-is;\n"
        "- 'clarify' meaning you should ask a short clarifying question to understand intent before searching;\n"
        "- 'expand' meaning the query contains multiple sub-questions and should be split into a list of focused sub-queries for retrieval.\n\n"
        f"User query: '''{user_query}'''\n\n"
        "Output a JSON object with keys: action, clarification, subqueries. If not applicable, set values to null or empty list.\n"
        "Be concise and only output valid JSON."
    )
    try:
        raw = llm(prompt)
        # try to parse JSON from model output
        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            # try to extract JSON-like substring
            m = re.search(r"(\{[\s\S]*\})", raw)
            if m:
                try:
                    parsed = json.loads(m.group(1))
                except Exception:
                    parsed = None
        if not parsed:
            # fallback heuristics
            q = user_query.strip()
            words = q.split()
            if len(words) <= 3 or not q.endswith('?'):
                return {"action": "clarify", "clarification": "Can you clarify what exactly you mean?", "subqueries": []}
            if len(words) > 25:
                return {"action": "expand", "clarification": None, "subqueries": [q]}
            return {"action": "direct", "clarification": None, "subqueries": []}
        # normalize
        action = parsed.get("action") if isinstance(parsed.get("action"), str) else parsed.get("action", "direct")
        clarification = parsed.get("clarification")
        subqueries = parsed.get("subqueries") or []
        return {"action": action, "clarification": clarification, "subqueries": subqueries}
    except Exception:
        # conservative fallback
        q = user_query.strip()
        if len(q.split()) > 25:
            return {"action": "expand", "clarification": None, "subqueries": [q]}
        if len(q.split()) <= 3:
            return {"action": "clarify", "clarification": "Can you be more specific about what you're looking for?", "subqueries": []}
        return {"action": "direct", "clarification": None, "subqueries": []}


def preprocess_query(user_query: str) -> str:
    """Clean and normalize the user query for efficient embedding and retrieval.

    Steps:
    - Trim and normalize whitespace
    - Remove repeated punctuation
    - Remove excessive length (truncate politely)
    - Strip salutations and signatures
    """
    q = user_query or ""
    q = q.strip()
    # remove common salutations at start/end
    q = re.sub(r'^(hi|hello|hey|dear)\b[:,\s]*', '', q, flags=re.I)
    q = re.sub(r'\b(thanks|thank you)[.!]*$', '', q, flags=re.I)
    # normalize whitespace
    q = re.sub(r"\s+", " ", q)
    # collapse repeated punctuation
    q = re.sub(r"([?.!,]){2,}", r"\1", q)
    # truncate if too long (keep ~600 chars)
    if len(q) > 600:
        q = q[:600].rsplit(' ', 1)[0] + '...'
    return q


def is_outside_org_query(user_query: str) -> Dict:
    """Use the LLM to detect whether the user's query is out-of-domain (not about the organization's stored data).

    Returns: {"outside": bool, "reason": str}
    """
    llm = ChatOpenAI(temperature=0)
    prompt = (
        f"You are a classifier that determines whether a user query is asking about information RELATED to the organization named '{ORG_NAME}' and the documents stored in its internal vector database.\n"
        "Answer in JSON with keys: outside (true/false) and reason (short).\n\n"
        f"User query: '''{user_query}'''\n\n"
        "If the query is about internal policies, staff, departments, projects, datasets, research, or other organization-specific content, set outside to false. Otherwise set outside to true."
    )
    try:
        raw = llm(prompt)
        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            m = re.search(r"(\{[\s\S]*\})", raw)
            if m:
                try:
                    parsed = json.loads(m.group(1))
                except Exception:
                    parsed = None
        if not parsed:
            # conservative default: treat short generic questions as outside
            q = user_query.strip()
            if len(q.split()) < 4:
                return {"outside": True, "reason": "Short/generic question — likely not organization-specific"}
            return {"outside": False, "reason": "Assumed organization-related"}
        return {"outside": bool(parsed.get("outside")), "reason": parsed.get("reason", "")}
    except Exception:
        q = user_query.strip()
        if len(q.split()) < 4:
            return {"outside": True, "reason": "Could not classify; treating as outside"}
        return {"outside": False, "reason": "Could not classify confidently"}


def suggest_org_followups(user_query: str, n: int = 3) -> List[str]:
    """Ask the LLM to propose short, organization-related follow-up questions the assistant can offer when the original query is out-of-domain."""
    llm = ChatOpenAI(temperature=0.2)
    prompt = (
        f"The user asked: '''{user_query}'''.\n\n"
        f"Generate {n} concise follow-up questions the user might want answered that ARE related to the organization '{ORG_NAME}' and its internal data. "
        "Return a JSON array of short strings."
    )
    try:
        raw = llm(prompt)
        arr = []
        try:
            arr = json.loads(raw)
            if not isinstance(arr, list):
                arr = []
        except Exception:
            m = re.search(r"(\[[\s\S]*\])", raw)
            if m:
                try:
                    arr = json.loads(m.group(1))
                except Exception:
                    arr = []
        if not arr:
            # fallback simple heuristics
            return [f"Tell me about {ORG_NAME}'s recent projects", f"Who are the key contacts at {ORG_NAME}?", f"What datasets or publications does {ORG_NAME} have?"]
        return arr[:n]
    except Exception:
        return [f"Tell me about {ORG_NAME}'s recent projects", f"Who are the key contacts at {ORG_NAME}?", f"What datasets or publications does {ORG_NAME} have?"]


def retrieve_for_query(user_query: str, k: int = 5, filters: Optional[Dict[str, List[str]]] = None) -> List[Dict]:
    """Embed and retrieve top documents across classes, returning list of hits (dicts).
    Each hit contains: class, text, source, category, distance
    """
    embedder = get_embedder()
    try:
        q_vector = embedder.embed_query(user_query)
    except Exception:
        return []

    classes = get_available_classes()
    if not classes:
        return []

    where_clause = build_where_clause(filters)
    all_hits = []
    for class_name in classes:
        gql = f'''{{ Get {{ {class_name}(nearVector: {{ vector: {q_vector} }} limit: {k}{where_clause}) {{ text source category _additional {{ distance }} }} }} }}'''
        try:
            r = requests.post(f"{WEAVIATE_URL}/v1/graphql", json={"query": gql}, timeout=10)
            r.raise_for_status()
            data = r.json()
            items = data.get("data", {}).get("Get", {}).get(class_name, [])
            for it in items:
                all_hits.append({
                    "class": class_name,
                    "text": it.get("text", ""),
                    "source": it.get("source", ""),
                    "category": it.get("category", class_name),
                    "distance": it.get("_additional", {}).get("distance", 0),
                })
        except Exception:
            continue
    all_hits.sort(key=lambda x: x.get("distance", 0))
    return all_hits[:k]


def build_where_clause(filters: Optional[Dict[str, List[str]]]) -> str:
    """Build a GraphQL 'where' clause string from filters mapping.
    filters example: {"category": ["department-of-computer-science"], "source": ["vision-and-mission"]}
    """
    if not filters:
        return ""
    operands = []
    for key, values in filters.items():
        for v in values:
            # escape quotes in v
            v_safe = v.replace('"', '\\"')
            operands.append(f'{{path: ["{key}"], operator: Equal, valueText: "{v_safe}"}}')
    if not operands:
        return ""
    if len(operands) == 1:
        return f", where: {operands[0]}"
    # combine with AND
    ops = ", ".join(operands)
    return f", where: {{ operator: And, operands: [ {ops} ] }}"


def perform_query(user_query: str, k: int = 5, filters: Optional[Dict[str, List[str]]] = None) -> str:
    """Main query function used by the agent.
    - If user_query is a greeting, respond directly using LLM (no retrieval).
    - Otherwise, embed the query and search Weaviate across available classes applying metadata filters.
    - Use ChatOpenAI to generate the final answer based on retrieved documents, ensuring organizational relevance.
    """
    # Preprocess the incoming query for normalization
    cleaned = preprocess_query(user_query)

    # Greeting shortcut
    if is_greeting(cleaned):
        llm = ChatOpenAI(temperature=0)
        prompt = f"You are a helpful assistant for {ORG_NAME}. Respond to this greeting briefly and politely: '{cleaned}'"
        return llm.call_as_llm(prompt) if hasattr(llm, 'call_as_llm') else llm(prompt)

    # Detect whether the question is about organization/internal data or outside scope
    outside_check = is_outside_org_query(cleaned)
    if outside_check.get("outside"):
        # Provide a brief generic answer, but clearly state it's outside org data
        llm = ChatOpenAI(temperature=0.2)
        gen_prompt = (
            f"The user asked: '{cleaned}'. This question is NOT about the organization's internal data. Provide a very brief, factual, and cautious general answer (max 3 sentences)."
        )
        try:
            general_answer = llm.call_as_llm(gen_prompt) if hasattr(llm, 'call_as_llm') else llm(gen_prompt)
        except Exception:
            general_answer = llm(gen_prompt)

        # Suggest organization-related follow-up questions the user may want instead
        suggestions = suggest_org_followups(cleaned, n=3)
        sugg_text = "\n".join([f"- {s}" for s in suggestions])
        return (
            f"I don't have organization-specific documents for that question. {outside_check.get('reason','')}\n\n"
            f"Brief general answer:\n{general_answer}\n\n"
            f"If you'd like, I can help with organization-related information. Here are some related questions you might want to ask:\n{sugg_text}\n\n"
            f"Which of these would you like information about?"
        )

    # Analyze intent: direct / clarify / expand
    intent = analyze_query_intent(cleaned)
    action = intent.get("action", "direct")

    if action == "clarify":
        return intent.get("clarification") or "Can you clarify what you mean?"

    # If expansion requested, obtain subqueries
    subqueries = intent.get("subqueries") or []
    if action == "expand" and not subqueries:
        # fallback: ask LLM to produce up to 5 focused sub-queries
        llm = ChatOpenAI(temperature=0)
        split_prompt = (
            "Split the following user query into up to 5 focused sub-questions suitable for document retrieval. "
            "Return a JSON array of short sub-questions.\n\n"
            f"Query: '''{cleaned}'''\n"
        )
        raw = llm(split_prompt)
        try:
            subqueries = json.loads(raw)
            if not isinstance(subqueries, list):
                subqueries = [cleaned]
        except Exception:
            subqueries = [user_query]

    # If we have multiple subqueries, run retrieval per subquery and synthesize
    if action == "expand" and len(subqueries) > 0:
        llm = ChatOpenAI(temperature=0)
        per_results = []
        for sq in subqueries:
            hits = retrieve_for_query(sq, k=k, filters=filters)
            if not hits:
                per_results.append({"subquery": sq, "answer": "No relevant documents found."})
                continue
            # build context for this subquery
            context_parts = [f"[Source: {h.get('source') or 'unknown'} | Category: {h.get('category')}]\n{h.get('text')}\n" for h in hits]
            context = "\n---\n".join(context_parts)
            system_prompt = (
                f"You are an assistant for {ORG_NAME}. Use ONLY the retrieved documents to answer the sub-question. "
                "Be concise and cite sources when possible."
            )
            user_prompt = f"Sub-question: {sq}\n\nRetrieved documents:\n{context}\n\nAnswer the sub-question based on the documents."
            try:
                sub_resp = llm.call_as_llm(system_prompt + "\n\n" + user_prompt) if hasattr(llm, 'call_as_llm') else llm(system_prompt + "\n\n" + user_prompt)
            except Exception:
                sub_resp = llm(system_prompt + "\n\n" + user_prompt)
            per_results.append({"subquery": sq, "answer": sub_resp})

        # Combine per-subquery answers into final chained response
        combined = []
        for pr in per_results:
            combined.append(f"Sub-question: {pr['subquery']}\nAnswer:\n{pr['answer']}\n")
        return "\n\n".join(combined)

    # Default: direct retrieval
    hits = retrieve_for_query(cleaned, k=k, filters=filters)
    if not hits:
        return "No relevant documents found."

    context_parts = []
    for h in hits:
        src = h.get("source") or "unknown"
        cat = h.get("category") or h.get("class")
        txt = h.get("text")
        context_parts.append(f"[Source: {src} | Category: {cat}]\n{txt}\n")
    context = "\n---\n".join(context_parts)

    system_prompt = (
        f"You are an assistant for {ORG_NAME}. Use ONLY the retrieved documents to answer the user's question. "
        "If the answer is not contained in the retrieved documents, say you don't know or suggest where to find official information. "
        "Be concise and factual, and cite the source and category for each fact when possible."
    )

    user_prompt = (
        f"User question: {cleaned}\n\nRetrieved documents:\n{context}\n\nAnswer the question based on the documents above."
    )

    llm = ChatOpenAI(temperature=0)
    try:
        resp = llm.call_as_llm(system_prompt + "\n\n" + user_prompt) if hasattr(llm, 'call_as_llm') else llm(system_prompt + "\n\n" + user_prompt)
    except Exception:
        resp = llm(system_prompt + "\n\n" + user_prompt)

    return resp
