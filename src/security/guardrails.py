from typing import Dict


HIGH_RISK_KEYWORDS = [
    "treatment",
    "medicine",
    "drug",
    "dose",
    "dosage",
    "prescribe",
    "prescription",
    "what should i take",
    "which medicine should i take",
    "can i take",
    "should i take",
    "how much should i take",
    "what is the best treatment",
    "cure",
    "die",
    "dead"
]

MEDIUM_RISK_KEYWORDS = [
    "do i have cancer",
    "is this cancer",
    "is it malignant",
    "is it benign",
    "can this be cancer",
    "does this mean cancer",
    "what does my exam mean",
    "is this serious",
]


def evaluate_question_risk(question: str) -> Dict[str, str]:
    question_lower = question.lower().strip()

    for keyword in HIGH_RISK_KEYWORDS:
        if keyword in question_lower:
            return {
                "safe": False,
                "risk_level": "high",
                "action": "block",
                "reason": "Pergunta com solicitação de tratamento, medicação ou dosagem."
            }

    for keyword in MEDIUM_RISK_KEYWORDS:
        if keyword in question_lower:
            return {
                "safe": True,
                "risk_level": "medium",
                "action": "allow_with_warning",
                "reason": "Pergunta com interpretação clínica sensível."
            }

    return {
        "safe": True,
        "risk_level": "low",
        "action": "allow",
        "reason": "Pergunta informativa."
    }


def build_guardrail_response(risk_result: Dict[str, str]) -> str:
    if risk_result["action"] == "block":
        return (
            "I cannot provide medical treatment, prescription, or dosage guidance. "
            "Please consult a qualified healthcare professional."
        )

    if risk_result["action"] == "allow_with_warning":
        return (
            "This question involves sensitive medical interpretation. "
            "I can provide general informational support, but this does not replace professional medical evaluation."
        )

    return ""