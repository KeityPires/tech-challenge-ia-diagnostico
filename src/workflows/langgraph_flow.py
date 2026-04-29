from typing import TypedDict, List, Dict, Any, Optional
import sqlite3

from langgraph.graph import StateGraph, END

from src.security.guardrails import evaluate_question_risk, build_guardrail_response
from src.assistant.response_formatter import format_sources
from src.security.logging_system import log_interaction

from src.multimodal.video_processor import analyze_video
from src.multimodal.audio_processor import analyze_audio
from src.multimodal.multimodal_fusion import calculate_multimodal_risk
from src.multimodal.alert_generator import generate_alert


class AssistantState(TypedDict, total=False):
    question: str
    risk_level: str
    action: str
    reason: str

    docs: List[Any]
    context: str

    # novos campos
    patient_id: Optional[str]
    patient_context: str
    final_context: str

    answer: str
    sources: List[Dict[str, Any]]
    status: str
    
    # novos campos - fase 4
    video_path: Optional[str]
    audio_path: Optional[str]

    video_result: Dict[str, Any]
    audio_result: Dict[str, Any]
    multimodal_result: Dict[str, Any]
    multimodal_context: str


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
        state["patient_context"] = ""
        state["final_context"] = ""
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


def patient_context_node(
    state: AssistantState,
    db_path: str = "data/medical_demo.db"
) -> AssistantState:
    patient_id = state.get("patient_id")

    # Se não vier patient_id, o fluxo continua normalmente
    if not patient_id:
        state["patient_context"] = "No patient data was provided."
        return state

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        patient = cursor.execute(
            """
            SELECT patient_id, name, age, sex
            FROM patients
            WHERE patient_id = ?
            """,
            (patient_id,)
        ).fetchone()

        encounters = cursor.execute(
            """
            SELECT visit_date, complaint, diagnosis, notes
            FROM encounters
            WHERE patient_id = ?
            ORDER BY visit_date DESC
            LIMIT 3
            """,
            (patient_id,)
        ).fetchall()

        exams = cursor.execute(
            """
            SELECT exam_date, exam_type, result
            FROM exams
            WHERE patient_id = ?
            ORDER BY exam_date DESC
            LIMIT 3
            """,
            (patient_id,)
        ).fetchall()

        conn.close()

        if not patient:
            state["patient_context"] = f"Patient {patient_id} was not found in the structured database."
            return state

        encounters_text = "\n".join(
            [
                f"- Date: {visit_date} | Complaint: {complaint} | Diagnosis: {diagnosis} | Notes: {notes}"
                for visit_date, complaint, diagnosis, notes in encounters
            ]
        ) if encounters else "- No recent encounters found."

        exams_text = "\n".join(
            [
                f"- Date: {exam_date} | Exam: {exam_type} | Result: {result}"
                for exam_date, exam_type, result in exams
            ]
        ) if exams else "- No recent exams found."

        state["patient_context"] = f"""
Structured patient data:
Patient: {patient[1]} (ID: {patient[0]})
Age: {patient[2]}
Sex: {patient[3]}

Recent encounters:
{encounters_text}

Recent exams:
{exams_text}
""".strip()

    except Exception as e:
        state["patient_context"] = f"Structured patient data could not be retrieved: {str(e)}"

    return state

def video_analysis_node(state: AssistantState) -> AssistantState:
    video_path = state.get("video_path")

    if not video_path:
        state["video_result"] = {
            "modality": "video",
            "risk_score": 0,
            "risk_level": "not_provided",
            "flags": [],
            "metadata": {}
        }
        return state

    state["video_result"] = analyze_video(video_path)
    return state


def audio_analysis_node(state: AssistantState) -> AssistantState:
    audio_path = state.get("audio_path")

    if not audio_path:
        state["audio_result"] = {
            "modality": "audio",
            "risk_score": 0,
            "risk_level": "not_provided",
            "flags": [],
            "transcription": ""
        }
        return state

    state["audio_result"] = analyze_audio(audio_path)
    return state


def multimodal_fusion_node(state: AssistantState) -> AssistantState:
    video_result = state.get("video_result", {})
    audio_result = state.get("audio_result", {})

    multimodal_result = calculate_multimodal_risk(video_result, audio_result)
    alert_text = generate_alert(multimodal_result)

    state["multimodal_result"] = multimodal_result
    state["multimodal_context"] = f"""
MULTIMODAL ANALYSIS:
Video risk score: {multimodal_result.get("video_score")}
Audio risk score: {multimodal_result.get("audio_score")}
Final multimodal score: {multimodal_result.get("final_score")}
Risk level: {multimodal_result.get("risk_level")}
Alert generated: {multimodal_result.get("alert")}

Evidence:
{chr(10).join(["- " + item for item in multimodal_result.get("evidences", [])])}

Alert message:
{alert_text}
""".strip()

    return state

def merge_context_node(state: AssistantState) -> AssistantState:
    patient_context = state.get("patient_context", "")
    rag_context = state.get("context", "")
    multimodal_context = state.get("multimodal_context", "")

    state["final_context"] = f"""
STRUCTURED PATIENT DATA:
{patient_context}

MULTIMODAL ANALYSIS:
{multimodal_context}

MEDICAL KNOWLEDGE RETRIEVED:
{rag_context}
""".strip()

    return state


def generate_node(state: AssistantState, llm) -> AssistantState:
    prompt = f"""
You are a medical educational assistant focused on women's health, maternal health, psychological well-being and breast cancer information.

Use only the context below to answer the question.
Prioritize patient-specific structured data when relevant.
Do not invent information.
If the context is insufficient, clearly say that the available sources are insufficient.
Do not provide a definitive diagnosis.
Do not prescribe treatment.
Do not prescribe dosage.
Always answer in a clear and educational tone.

Use multimodal analysis only as triage support, never as a definitive diagnosis.

Context:
{state.get("final_context", state.get("context", ""))}

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
        answer=state.get("answer", ""),
        sources=state.get("sources", []),
        risk_level=state.get("risk_level", ""),
        status=state.get("status", ""),
        retrieved_docs_count=len(state["docs"]) if state.get("docs") else 0
    )
    return state


def build_medical_assistant_graph(llm, retriever, db_path: str = "data/medical_demo.db"):
    graph = StateGraph(AssistantState)

    graph.add_node("guardrails", guardrails_node)
    graph.add_node("retrieve", lambda state: retrieve_node(state, retriever))
    graph.add_node("patient_context", lambda state: patient_context_node(state, db_path))
    graph.add_node("video_analysis", video_analysis_node)
    graph.add_node("audio_analysis", audio_analysis_node)
    graph.add_node("multimodal_fusion", multimodal_fusion_node)
    graph.add_node("merge_context", merge_context_node)
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

    graph.add_edge("retrieve", "patient_context")
    graph.add_edge("patient_context", "video_analysis")
    graph.add_edge("video_analysis", "audio_analysis")
    graph.add_edge("audio_analysis", "multimodal_fusion")
    graph.add_edge("multimodal_fusion", "merge_context")
    graph.add_edge("merge_context", "generate")
    graph.add_edge("generate", "format")
    graph.add_edge("format", "log")
    graph.add_edge("log", END)

    return graph.compile()