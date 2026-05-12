def calculate_multimodal_risk(video_result: dict, audio_result: dict) -> dict:
    video_score = float(video_result.get("risk_score", 0) or 0)
    audio_score = float(audio_result.get("risk_score", 0) or 0)

    video_level = video_result.get("risk_level", "not_provided")
    audio_level = audio_result.get("risk_level", "not_provided")

    video_flags = video_result.get("flags", []) or []
    audio_flags = audio_result.get("flags", []) or []

    evidences = list(dict.fromkeys(video_flags + audio_flags))

    has_video = video_level != "not_provided" and video_score > 0
    has_audio = audio_level != "not_provided" and audio_score > 0

    if has_video and has_audio:
        final_score = (0.4 * video_score) + (0.6 * audio_score)
        fusion_strategy = "audio_60_video_40"
    elif has_audio:
        final_score = audio_score
        fusion_strategy = "audio_only"
    elif has_video:
        final_score = video_score
        fusion_strategy = "video_only"
    else:
        final_score = 0
        fusion_strategy = "no_multimodal_data"

    final_score = round(final_score, 2)

    # Regras de reforço para sinais emocionalmente relevantes
    high_risk_flags = {
        "fear_expression",
        "angry_expression",
        "sad_expression",
        "confused_expression",
        "ansiosa",
        "agitada",
        "cansada",
        "fadiga_hormonal_ou_cansaco",
        "violencia_domestica",
        "trauma",
        "medo",
        "choro",
        "hesitacao",
        "persistent_fear",
        "persistent_sadness",
        "persistent_tension",
        "persistent_confusion",
        "emotional_variation_detected",
        "face_not_visible",
    }

    relevant_signals = [
        flag for flag in evidences
        if flag in high_risk_flags
    ]

    if final_score >= 0.7:
        risk_level = "alto"
    elif final_score >= 0.4:
        risk_level = "medio"
    elif final_score > 0:
        risk_level = "baixo"
    else:
        risk_level = "not_provided"

    # se houver muitos sinais relevantes mesmo com score moderado, eleva para médio
    if risk_level == "baixo" and len(relevant_signals) >= 3:
        risk_level = "medio"

    # Alerta apenas para alto ou médio com múltiplas evidências
    alert = risk_level == "alto" or (
        risk_level == "medio" and len(relevant_signals) >= 4
    )

    interpretation = []

    if not has_video and not has_audio:
        interpretation.append(
            "Não foram fornecidos dados multimodais suficientes para avaliação."
        )

    if has_audio:
        if audio_score >= 0.6:
            interpretation.append(
                "A análise de áudio apresentou sinais relevantes de possível ansiedade, agitação ou fadiga emocional."
            )
        elif audio_score >= 0.4:
            interpretation.append(
                "A análise de áudio apresentou sinais moderados que podem indicar desconforto emocional."
            )
        else:
            interpretation.append(
                "A análise de áudio apresentou baixo nível de risco."
            )

    if has_video:
        if video_score >= 0.6:
            interpretation.append(
                "A análise de vídeo apresentou sinais visuais relevantes de possível desconforto emocional."
            )
        elif video_score >= 0.3:
            interpretation.append(
                "A análise de vídeo apresentou sinais visuais leves ou moderados, usados como evidência complementar."
            )
        else:
            interpretation.append(
                "A análise de vídeo apresentou baixo nível de risco visual."
            )

    if "fear_expression" in evidences:
        interpretation.append(
            "Foram observadas expressões aparentes associadas a medo, que devem ser interpretadas com cautela."
        )

    if "sad_expression" in evidences:
        interpretation.append(
            "Foram observadas expressões aparentes associadas a tristeza."
        )

    if "angry_expression" in evidences:
        interpretation.append(
            "Foram observadas expressões aparentes associadas a tensão ou raiva."
        )

    if "confused_expression" in evidences:
        interpretation.append(
            "Foram observadas expressões aparentes associadas a confusão ou insegurança."
        )

    if "ansiosa" in evidences or "agitada" in evidences:
        interpretation.append(
            "O áudio indicou sinais compatíveis com ansiedade ou agitação."
        )

    if "cansada" in evidences or "fadiga_hormonal_ou_cansaco" in evidences:
        interpretation.append(
            "O áudio indicou sinais compatíveis com cansaço ou fadiga."
        )

    recommendation = ""

    if risk_level == "alto":
        recommendation = (
            "Recomenda-se priorizar avaliação por profissional de saúde, "
            "especialmente se os sinais forem persistentes, intensos ou associados a sofrimento."
        )
    elif risk_level == "medio":
        recommendation = (
            "Recomenda-se acompanhamento e nova avaliação clínica caso os sinais persistam, "
            "se intensifiquem ou estejam associados a sofrimento emocional."
        )
    elif risk_level == "baixo":
        recommendation = (
            "Não foram identificados sinais críticos, mas recomenda-se manter observação "
            "e buscar orientação profissional em caso de agravamento."
        )
    else:
        recommendation = (
            "Não há dados multimodais suficientes para gerar uma recomendação específica."
        )

    return {
        "final_score": final_score,
        "risk_level": risk_level,
        "alert": alert,
        "evidences": evidences,
        "relevant_signals": relevant_signals,
        "video_score": video_score,
        "audio_score": audio_score,
        "video_risk_level": video_level,
        "audio_risk_level": audio_level,
        "fusion_strategy": fusion_strategy,
        "interpretation": interpretation,
        "recommendation": recommendation,
        "limitations": [
            "A análise multimodal é apenas apoio à triagem clínica.",
            "O sistema não realiza diagnóstico médico, psicológico ou psiquiátrico.",
            "Expressões faciais indicam apenas emoções aparentes.",
            "Sinais de áudio e vídeo devem ser interpretados como evidências complementares.",
            "A confirmação clínica depende de avaliação profissional."
        ]
    }