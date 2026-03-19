from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END

from src.security.guardrails import evaluate_question_risk, build_guardrail_response
from src.assistant.response_formatter import format_sources
from src.security.logging_system import log_interaction


class AssistantState(TypedDict):
    question: str
    risk_level: str
    action: str
    reason: str
    docs: List[Any]
    context: str
    answer: str
    sources: List[Dict[str, Any]]
    status: str


def guardrails_node(state: AssistantState) -> AssistantState:
    question = state["question"]
    risk = evaluate_question_risk(question)

    state["risk_level"] = risk["risk_level"]
    state["action"] = risk["action"]
    state["reason"] = risk["reason"]

    if risk["action"] == "block":
        state["answer"] = build_guardrail_response(risk)
        state["sources"] = []
        state["docs"] = []
        state["context"] = ""
        state["status"] = "blocked"

    return state


def guardrails_router(state: AssistantState) -> str:
    if state["action"] == "block":
        return "log"
    return "retrieve"


def retrieve_node(state: AssistantState, retriever) -> AssistantState:
    docs = retriever.invoke(state["question"])
    context = "\n".join([doc.page_content for doc in docs])

    state["docs"] = docs
    state["context"] = context
    return state


def generate_node(state: AssistantState, llm) -> AssistantState:
    prompt = f"""
You are a medical educational assistant focused on breast cancer information.

Use only the context below to answer the question.
Do not invent information.
If the context is insufficient, clearly say that the available sources are insufficient.
Do not provide a definitive diagnosis.
Do not prescribe treatment.
Do not prescribe dosage.
Always answer in a clear and educational tone.

Context:
{state["context"]}

Question:
{state["question"]}
"""

    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    if state["action"] == "allow_with_warning":
        warning = build_guardrail_response({
            "action": state["action"],
            "risk_level": state["risk_level"],
            "reason": state["reason"]
        })
        answer = f"{warning}\n\n{answer}"

    state["answer"] = answer
    state["status"] = "success"
    return state


def format_node(state: AssistantState) -> AssistantState:
    sources = format_sources(state["docs"])
    state["sources"] = sources
    return state


def log_node(state: AssistantState) -> AssistantState:
    log_interaction(
        question=state["question"],
        answer=state["answer"],
        sources=state["sources"],
        risk_level=state["risk_level"],
        status=state["status"],
        retrieved_docs_count=len(state["docs"]) if state["docs"] else 0
    )
    return state


def build_medical_assistant_graph(llm, retriever):
    graph = StateGraph(AssistantState)

    graph.add_node("guardrails", guardrails_node)
    graph.add_node("retrieve", lambda state: retrieve_node(state, retriever))
    graph.add_node("generate", lambda state: generate_node(state, llm))
    graph.add_node("format", format_node)
    graph.add_node("log", log_node)

    graph.set_entry_point("guardrails")

    graph.add_conditional_edges(
        "guardrails",
        guardrails_router,
        {
            "retrieve": "retrieve",
            "log": "log"
        }
    )

    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "format")
    graph.add_edge("format", "log")
    graph.add_edge("log", END)

    return graph.compile()