from src.security.guardrails import evaluate_question_risk, build_guardrail_response
from src.assistant.response_formatter import format_sources


def answer_medical_question(question: str, llm, retriever):
    
    # 1. Avaliar risco
    risk = evaluate_question_risk(question)

    # 2. Se for alto risco → bloquear
    if risk["action"] == "block":
        return {
            "answer": build_guardrail_response(risk),
            "sources": [],
            "status": "blocked",
            "risk_level": risk["risk_level"]
        }

    # 3. Recuperar contexto
    docs = retriever.invoke(question)

    context = "\n".join([doc.page_content for doc in docs])

    # 4. Prompt
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
{context}

Question:
{question}
"""

    # 5. LLM
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    # 6. Warning (se médio risco)
    warning = ""
    if risk["action"] == "allow_with_warning":
        warning = build_guardrail_response(risk) + "\n\n"

    final_answer = f"{warning}{answer}"

    # 7. Fontes
    sources = format_sources(docs)

    return {
        "answer": final_answer,
        "sources": sources,
        "status": "success",
        "risk_level": risk["risk_level"]
    }