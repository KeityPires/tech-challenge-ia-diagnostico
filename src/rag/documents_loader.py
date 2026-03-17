import json
from langchain_core.documents import Document


def load_medquad_documents(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []

    for item in data:
        content = f"Question: {item['question']}\nAnswer: {item['answer']}"

        metadata = {
            "id": item.get("id"),
            "source_file": item.get("source_file"),
            "collection": item.get("collection"),
            "source": "MedQuAD"
        }

        documents.append(
            Document(
                page_content=content,
                metadata=metadata
            )
        )

    return documents