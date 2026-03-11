from typing import TypedDict
from langgraph.graph import StateGraph, END

class AssistantState(TypedDict):
    question: str
    safe: bool
    retrieved_context: list
    answer: str
    sources: list

def check_safety_node(state):
    question = state["question"]
    forbidden = ["prescreva", "dose exata", "diagnóstico definitivo"]
    safe = not any(term in question.lower() for term in forbidden)
    state["safe"] = safe
    return state

def retrieve_node(state, vector_store):
    if not state["safe"]:
        state["retrieved_context"] = []
        return state

    docs = vector_store.similarity_search(state["question"], k=3)
    state["retrieved_context"] = docs
    return state

def answer_node(state, llm):
    if not state["safe"]:
        state["answer"] = (
            "Solicitação bloqueada por regra de segurança. "
            "É necessária validação humana."
        )
        state["sources"] = []
        return state

    docs = state["retrieved_context"]
    context = "\n\n".join(
        [f"Fonte: {doc.metadata.get('source', 'desconhecida')}\n{doc.page_content}" for doc in docs]
    )

    prompt = f"""
Pergunta: {state['question']}

Contexto:
{context}

Responda com cautela, sem extrapolar além do contexto.
Informe as fontes no final.
"""

    response = llm.invoke(prompt)
    state["answer"] = response.content
    state["sources"] = [doc.metadata.get("source", "desconhecida") for doc in docs]
    return state

def build_graph(vector_store, llm):
    graph = StateGraph(AssistantState)

    graph.add_node("check_safety", check_safety_node)
    graph.add_node("retrieve", lambda state: retrieve_node(state, vector_store))
    graph.add_node("answer", lambda state: answer_node(state, llm))

    graph.set_entry_point("check_safety")
    graph.add_edge("check_safety", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)

    return graph.compile()