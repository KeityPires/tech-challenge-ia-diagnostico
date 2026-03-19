import os
import json
from datetime import datetime

LOG_PATH = "data/assistant_logs.jsonl"


def log_interaction(
    question: str,
    answer: str,
    sources: list,
    risk_level: str,
    status: str,
    retrieved_docs_count: int
):

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "risk_level": risk_level,
        "status": status,
        "retrieved_docs_count": retrieved_docs_count,
        "sources": sources,
        "answer_preview": answer[:300],  # evita log gigante
        "model": "mistral",
        "collection": "CancerGov"
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")