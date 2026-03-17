from langchain_core.messages import SystemMessage, HumanMessage
from src.assistant.prompts import SYSTEM_PROMPT
from src.security.guardrails import is_safe_question
from src.security.logging_system import log_interaction


def format_sources(docs):
    sources = []

    for i, doc in enumerate(docs, start=1):
        sources.append(
            {
                "label": f"Fonte {i}",
                "source": doc.metadata.get("source", "desconhecida"),
                "collection": doc.metadata.get("collection", "desconhecida"),
                "source_file": doc.metadata.get("source_file", "desconhecido"),
                "id": doc.metadata.get("id", "sem_id")
            }
        )

    return sources


def answer_medical_question(question, llm, retriever):
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

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [
            f"Fonte: {doc.metadata.get('source_file', 'desconhecida')}\n{doc.page_content}"
            for doc in docs
        ]
    )

    user_prompt = f"""
Pergunta clínica:
{question}

Contexto recuperado:
{context}

Responda de forma objetiva, cautelosa e baseada apenas no contexto.
Se não houver contexto suficiente, diga isso explicitamente.
"""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)
    answer = response.content
    sources = format_sources(docs)

    log_interaction(question, answer, sources)

    return {
        "answer": answer,
        "sources": sources
    }