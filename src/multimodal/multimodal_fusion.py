def calculate_multimodal_risk(video_result: dict, audio_result: dict) -> dict:
    video_score = video_result.get("risk_score", 0)
    audio_score = audio_result.get("risk_score", 0)

    final_score = (0.4 * video_score) + (0.6 * audio_score)

    if final_score >= 0.7:
        risk_level = "alto"
    elif final_score >= 0.4:
        risk_level = "medio"
    else:
        risk_level = "baixo"

    return {
        "final_score": round(final_score, 2),
        "risk_level": risk_level,
        "alert": risk_level == "alto",
        "evidences": video_result.get("flags", []) + audio_result.get("flags", []),
        "video_score": video_score,
        "audio_score": audio_score
    }