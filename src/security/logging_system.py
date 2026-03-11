import json
from datetime import datetime
from src.config import LOG_PATH

def log_interaction(question: str, answer: str, sources: list):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "answer": answer,
        "sources": sources
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")