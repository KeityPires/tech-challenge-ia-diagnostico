FORBIDDEN_PATTERNS = [
    "prescreva",
    "dose exata",
    "substitua o médico",
    "diagnóstico definitivo"
]

def is_safe_question(question: str) -> bool:
    question_lower = question.lower()
    return not any(pattern in question_lower for pattern in FORBIDDEN_PATTERNS)