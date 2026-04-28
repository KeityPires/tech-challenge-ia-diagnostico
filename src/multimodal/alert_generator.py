def generate_alert(risk_result: dict) -> str:
    if not risk_result.get("alert"):
        return (
            "Sem alerta crítico no momento. "
            "Recomenda-se acompanhamento regular pela equipe de saúde."
        )

    evidences = ", ".join(risk_result.get("evidences", []))

    return (
        f"ALERTA DE RISCO {risk_result['risk_level'].upper()}. "
        f"Score final: {risk_result['final_score']}. "
        f"Evidências identificadas: {evidences}. "
        "Recomenda-se avaliação por equipe médica especializada. "
        "Este sistema não realiza diagnóstico, apenas apoio à triagem."
    )