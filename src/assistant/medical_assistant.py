from langchain_core.messages import SystemMessage, HumanMessage
from src.assistant.prompts import SYSTEM_PROMPT
from src.security.guardrails import is_safe_question
from src.security.logging_system import log_interaction

def answer_medical_question(question, llm, vector_store):
    if not is_safe_question(question):
        answer = (
            "Não posso fornecer esse tipo de orientação diretamente. "
            "É necessária validação humana por profissional habilitado."
        )
        log_interaction(question, answer, [])
        return {
            "answer": answer,
            "sources": []
        }

    docs = vector_store.similarity_search(question, k=3)

    context = "\n\n".join(
        [f"Fonte: {doc.metadata.get('source', 'desconhecida')}\n{doc.page_content}" for doc in docs]
    )

    user_prompt = f"""
Pergunta clínica:
{question}

Contexto recuperado:
{context}

Responda de forma objetiva, cautelosa e baseada apenas no contexto.
Inclua ao final as fontes consultadas.
"""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)
    answer = response.content
    sources = [doc.metadata.get("source", "desconhecida") for doc in docs]

    log_interaction(question, answer, sources)

    return {
        "answer": answer,
        "sources": sources
    }