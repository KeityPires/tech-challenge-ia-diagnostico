from src.workflows.langgraph_flow import build_medical_assistant_graph


def answer_medical_question(
    question: str,
    llm,
    retriever,
    patient_id: str | None = None,
    video_path: str | None = None,
    audio_path: str | None = None,
    video_s3_key: str | None = None,
    audio_s3_key: str | None = None,
    audio_language: str = "pt-BR"
):
    app = build_medical_assistant_graph(
        llm=llm,
        retriever=retriever
    )

    initial_state = {
        "question": question,
        "risk_level": "",
        "action": "",
        "reason": "",

        "docs": [],
        "context": "",

        "answer": "",
        "sources": [],
        "status": "",

        "patient_id": patient_id,
        "patient_context": "",
        "final_context": "",

        "video_path": video_path,
        "audio_path": audio_path,

        "video_result": {},
        "audio_result": {},
        "multimodal_result": {},
        "multimodal_context": "",

        "video_s3_key": video_s3_key,
        "audio_s3_key": audio_s3_key,
        "audio_language": audio_language
    }

    result = app.invoke(initial_state)

    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "status": result.get("status", ""),
        "risk_level": result.get("risk_level", ""),
        "video_result": result.get("video_result", {}),
        "audio_result": result.get("audio_result", {}),
        "multimodal_result": result.get("multimodal_result", {}),
        "multimodal_context": result.get("multimodal_context", "")
    }