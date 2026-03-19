from src.workflows.langgraph_flow import build_medical_assistant_graph


def answer_medical_question(question: str, llm, retriever):
    app = build_medical_assistant_graph(llm=llm, retriever=retriever)

    initial_state = {
        "question": question,
        "risk_level": "",
        "action": "",
        "reason": "",
        "docs": [],
        "context": "",
        "answer": "",
        "sources": [],
        "status": ""
    }

    result = app.invoke(initial_state)

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "status": result["status"],
        "risk_level": result["risk_level"]
    }