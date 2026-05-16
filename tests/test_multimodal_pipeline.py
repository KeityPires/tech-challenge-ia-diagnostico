import pytest

from alert_generator import generate_alert
from audio_processor import (
    classify_emotional_categories,
    calculate_audio_risk,
    build_audio_interpretation,
)
from multimodal_fusion import calculate_multimodal_risk
from video_processor import extract_video_metadata

# TESTES: alert_generator.py

def test_generate_alert_without_critical_alert():
    risk_result = {
        "alert": False,
        "risk_level": "baixo",
        "final_score": 0.25,
        "evidences": []
    }

    result = generate_alert(risk_result)

    assert "Sem alerta crítico" in result
    assert "acompanhamento regular" in result


def test_generate_alert_with_high_risk():
    risk_result = {
        "alert": True,
        "risk_level": "alto",
        "final_score": 0.85,
        "evidences": ["fear_expression", "persistent_fear"]
    }

    result = generate_alert(risk_result)

    assert "ALERTA DE RISCO ALTO" in result
    assert "Score final: 0.85" in result
    assert "fear_expression" in result
    assert "persistent_fear" in result
    assert "não realiza diagnóstico" in result

# TESTES: audio_processor.py

def test_classify_emotional_categories_detects_anxiety_and_sleep_change():
    transcription = "Estou ansiosa, cansada e não consigo dormir."

    result = classify_emotional_categories(transcription)

    assert "ansiedade" in result["detected_categories"]
    assert "fadiga_hormonal_ou_cansaco" in result["detected_categories"]
    assert "alteracao_do_sono" in result["detected_categories"]
    assert "ansiosa" in result["flags"]
    assert "cansada" in result["flags"]


def test_classify_emotional_categories_without_alert_terms():
    transcription = "Estou tranquila hoje e me sentindo bem."

    result = classify_emotional_categories(transcription)

    assert result["detected_categories"] == {}
    assert result["flags"] == []


def test_calculate_audio_risk_with_only_weak_acoustic_flags_is_limited():
    detected_categories = {}
    voice_features = {
        "tone_flags": [
            "voice_instability",
            "elevated_voice_tension",
            "speech_hesitation",
        ],
        "voice_intensity": "alta"
    }

    score = calculate_audio_risk(detected_categories, voice_features)

    assert score <= 0.35


def test_calculate_audio_risk_with_violence_or_fear_terms():
    detected_categories = {
        "sinais_de_violencia_ou_medo": ["tenho medo", "machucada"]
    }
    voice_features = {
        "tone_flags": [],
        "voice_intensity": "moderada"
    }

    score = calculate_audio_risk(detected_categories, voice_features)

    assert score >= 0.45


def test_build_audio_interpretation_without_textual_alert_but_with_acoustic_flags():
    detected_categories = {}
    flags = ["voice_instability", "speech_hesitation"]
    voice_features = {
        "estimated_voice_profile": "feminina",
        "voice_intensity": "moderada",
        "pitch_variation": 85,
        "silence_ratio": 0.4,
        "tone_flags": ["voice_instability", "speech_hesitation"]
    }

    interpretation = build_audio_interpretation(
        detected_categories,
        flags,
        voice_features
    )

    assert "voz feminina" in interpretation
    assert "Não foram identificadas palavras-chave clínicas" in interpretation
    assert "evidências complementares de baixa confiança" in interpretation
    assert "não representam diagnóstico" in interpretation

# TESTES: multimodal_fusion.py

def test_calculate_multimodal_risk_audio_only_low_risk():
    video_result = {
        "risk_score": 0,
        "risk_level": "not_provided",
        "flags": []
    }

    audio_result = {
        "risk_score": 0.25,
        "risk_level": "baixo",
        "flags": [
            "voice_instability",
            "speech_hesitation"
        ]
    }

    result = calculate_multimodal_risk(video_result, audio_result)

    assert result["fusion_strategy"] == "audio_only"
    assert result["risk_level"] == "baixo"
    assert result["alert"] is False
    assert result["final_score"] == 0.25
    assert "voice_instability" not in result["display_evidences"]


def test_calculate_multimodal_risk_video_only_high_risk():
    video_result = {
        "risk_score": 0.9,
        "risk_level": "alto",
        "flags": [
            "person_detected",
            "face_detected",
            "fear_expression",
            "persistent_fear"
        ]
    }

    audio_result = {
        "risk_score": 0,
        "risk_level": "not_provided",
        "flags": []
    }

    result = calculate_multimodal_risk(video_result, audio_result)

    assert result["fusion_strategy"] == "video_only"
    assert result["risk_level"] == "alto"
    assert result["alert"] is True
    assert result["video_score"] == 0.9
    assert "fear_expression" in result["evidences"]


def test_calculate_multimodal_risk_audio_60_video_40():
    video_result = {
        "risk_score": 0.9,
        "risk_level": "alto",
        "flags": ["fear_expression", "persistent_fear"]
    }

    audio_result = {
        "risk_score": 0.8,
        "risk_level": "alto",
        "flags": ["ansiosa", "cansada"]
    }

    result = calculate_multimodal_risk(video_result, audio_result)

    expected_score = round((0.4 * 0.9) + (0.6 * 0.8), 2)

    assert result["fusion_strategy"] == "audio_60_video_40"
    assert result["final_score"] == expected_score
    assert result["risk_level"] == "alto"
    assert result["alert"] is True


def test_calculate_multimodal_risk_without_data():
    video_result = {
        "risk_score": 0,
        "risk_level": "not_provided",
        "flags": []
    }

    audio_result = {
        "risk_score": 0,
        "risk_level": "not_provided",
        "flags": []
    }

    result = calculate_multimodal_risk(video_result, audio_result)

    assert result["fusion_strategy"] == "no_multimodal_data"
    assert result["risk_level"] == "not_provided"
    assert result["alert"] is False
    assert result["final_score"] == 0

# TESTES: video_processor.py

def test_extract_video_metadata_file_not_found():
    with pytest.raises(FileNotFoundError):
        extract_video_metadata("arquivo_inexistente.mp4")