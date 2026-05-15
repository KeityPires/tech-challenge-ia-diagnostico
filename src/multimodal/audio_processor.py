from pathlib import Path
import wave
import audioop
import speech_recognition as sr
import librosa
import numpy as np


def get_voice_features(audio_path: str) -> dict:
    audio_file = Path(audio_path)

    try:
        y, sr_rate = librosa.load(str(audio_file), sr=None)

        duration_seconds = librosa.get_duration(y=y, sr=sr_rate)

        rms = librosa.feature.rms(y=y)[0]
        mean_energy = float(np.mean(rms))

        pitches, magnitudes = librosa.piptrack(y=y, sr=sr_rate)
        pitch_values = pitches[pitches > 0]
        pitch_variation = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0

        mean_pitch = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0

        if mean_pitch >= 165:
            estimated_voice_profile = "feminina"
        elif mean_pitch > 0:
            estimated_voice_profile = "masculina"
        else:
            estimated_voice_profile = "indefinido"

        non_silent_intervals = librosa.effects.split(y, top_db=20)
        non_silent_samples = sum(end - start for start, end in non_silent_intervals)
        silence_ratio = 1 - (non_silent_samples / len(y)) if len(y) > 0 else 0

        if mean_energy > 0.05:
            voice_intensity = "alta"
        elif mean_energy > 0.02:
            voice_intensity = "moderada"
        else:
            voice_intensity = "baixa"

        tone_flags = []

        if pitch_variation > 70:
            tone_flags.append("voice_instability")

        if mean_pitch > 220:
            tone_flags.append("elevated_voice_tension")

        if silence_ratio > 0.35:
            tone_flags.append("speech_hesitation")

        if mean_energy < 0.02:
            tone_flags.append("low_voice_energy")

        return {
            "duration_seconds": round(duration_seconds, 2),
            "mean_energy": round(mean_energy, 4),
            "mean_pitch": round(mean_pitch, 2),
            "pitch_variation": round(pitch_variation, 2),
            "silence_ratio": round(float(silence_ratio), 2),
            "voice_intensity": voice_intensity,
            "tone_flags": tone_flags,
            "estimated_voice_profile": estimated_voice_profile,
        }

    except Exception as e:
        return {
            "duration_seconds": None,
            "mean_energy": None,
            "mean_pitch": None,
            "pitch_variation": None,
            "silence_ratio": None,
            "voice_intensity": "indisponível",
            "tone_flags": [],
            "error": str(e)
        }

def classify_emotional_categories(transcription: str) -> dict:
    text = transcription.lower()

    emotional_categories = {
        "ansiedade": [
            "ansiosa", "ansioso", "ansiedade", "agitada", "agitado",
            "nervosa", "nervoso", "angústia", "angustia", "preocupada", 
            "medo"
        ],
        "depressao_pos_parto_ou_sofrimento_emocional": [
            "sem vontade", "não tenho vontade", "nao tenho vontade",
            "não quero levantar", "nao quero levantar", "fico na cama",
            "triste", "muito mal", "desanimada", "sem energia", 
            "indisposta", "vazio", "apatica", "apática"
        ],
        "sinais_de_violencia_ou_medo": [
            "medo", "ameaça", "ameaca", "machucada", "machucado",
            "não posso falar", "nao posso falar", "ele não deixa",
            "ela não deixa", "tenho medo", "melhor eu ir", "não quero responder", "dor",
            "fissura", "andência", "incômodo", "exagerando", "minha culpa", "culpa"
        ],
        "fadiga_hormonal_ou_cansaco": [
            "cansaço", "cansaco", "cansada", "cansado",
            "exausta", "exausto", "fadiga", "sono", "sem energia"
        ],
        "alteracao_do_sono": [
            "não consigo dormir", "nao consigo dormir",
            "insônia", "insonia", "dificuldade para dormir",
            "durmo mal", "acordo muito", "acordo muitas vezes", "sono superficial"
        ]
    }

    detected_categories = {}
    found_flags = []

    for category, terms in emotional_categories.items():
        matches = [term for term in terms if term in text]

        if matches:
            detected_categories[category] = matches
            found_flags.extend(matches)

    return {
        "detected_categories": detected_categories,
        "flags": list(dict.fromkeys(found_flags))
    }


def calculate_audio_risk(detected_categories: dict, voice_features: dict) -> float:
    category_count = len(detected_categories)

    score = 0.0

    if category_count > 0:
        score += 0.20 + (0.12 * category_count)

    if "sinais_de_violencia_ou_medo" in detected_categories:
        score += 0.25

    if "depressao_pos_parto_ou_sofrimento_emocional" in detected_categories:
        score += 0.20

    tone_flags = voice_features.get("tone_flags", [])

    # Sinais acústicos isolados devem ter peso menor
    if "voice_instability" in tone_flags:
        score += 0.07

    if "elevated_voice_tension" in tone_flags:
        score += 0.07

    if "speech_hesitation" in tone_flags:
        score += 0.06

    if "low_voice_energy" in tone_flags:
        score += 0.04

    if voice_features.get("voice_intensity") == "alta":
        score += 0.05

    # Se não houve nenhuma categoria textual, limita o score acústico
    if category_count == 0:
        score = min(score, 0.35)

    return min(score, 1.0)

def build_audio_interpretation(detected_categories: dict, flags: list, voice_features: dict) -> str:

    interpretation_parts = []

    voice_profile = voice_features.get(
        "estimated_voice_profile",
        "indefinido"
    )

    voice_intensity = voice_features.get(
        "voice_intensity",
        "indefinida"
    )

    pitch_variation = voice_features.get(
        "pitch_variation",
        0
    )

    silence_ratio = voice_features.get(
        "silence_ratio",
        0
    )

    tone_flags = voice_features.get(
        "tone_flags",
        []
    )

    has_textual_categories = bool(detected_categories)
    has_acoustic_flags = bool(tone_flags)

    # perfil vocal estimado
    if voice_profile == "feminina":
        interpretation_parts.append(
            "O perfil vocal estimado é compatível com voz feminina."
        )

    elif voice_profile == "masculina":
        interpretation_parts.append(
            "O perfil vocal estimado é compatível com voz masculina."
        )

    else:
        interpretation_parts.append(
            "Não foi possível estimar o perfil vocal com segurança."
        )

    # categorias emocionais pela transcrição
    if has_textual_categories:
        categories_text = ", ".join(
            detected_categories.keys()
        )

        interpretation_parts.append(
            f"A análise textual identificou categorias associadas a {categories_text}."
        )

        flags_text = ", ".join(flags)

        if flags_text:
            interpretation_parts.append(
                f"Os principais indicadores textuais e vocais encontrados foram: {flags_text}."
            )

    else:
        interpretation_parts.append(
            "Não foram identificadas palavras-chave clínicas de alerta na transcrição."
        )

    # aviso específico para bases como RAVDESS
    if not has_textual_categories and has_acoustic_flags:
        interpretation_parts.append(
            "Os sinais observados vieram principalmente de características acústicas da voz, "
            "como variação de pitch, pausas, energia ou intensidade vocal. "
            "Como a transcrição não contém termos clínicos de alerta, esses sinais devem ser tratados "
            "como evidências complementares de baixa confiança."
        )

    # intensidade geral
    interpretation_parts.append(
        f"A intensidade vocal foi classificada como {voice_intensity}."
    )

    # sinais vocais com linguagem mais cautelosa
    if "voice_instability" in flags:
        interpretation_parts.append(
            "Foram registradas oscilações vocais, que podem indicar variação emocional, "
            "mas isoladamente não confirmam ansiedade ou tensão."
        )

    if "speech_hesitation" in flags:
        interpretation_parts.append(
            "Foram identificadas pausas ou hesitações na fala, interpretadas apenas como sinal acústico complementar."
        )

    if "elevated_voice_tension" in flags:
        interpretation_parts.append(
            "A análise acústica registrou possível tensão vocal, sem confirmação clínica isolada."
        )

    if "low_voice_energy" in flags:
        interpretation_parts.append(
            "A energia vocal reduzida pode estar relacionada a calma, cansaço ou baixa intensidade de fala, "
            "não sendo suficiente para indicar fadiga clínica isoladamente."
        )

    # variação vocal
    if pitch_variation and pitch_variation > 70:
        interpretation_parts.append(
            "A variação do pitch foi elevada, sendo considerada um sinal acústico complementar, "
            "mas não deve ser interpretada isoladamente como ansiedade."
        )

    # pausas/silêncio
    if silence_ratio and silence_ratio > 0.35:
        interpretation_parts.append(
            "A proporção de silêncio ou pausas foi elevada, podendo refletir ritmo de fala, gravação, "
            "interpretação do ator ou hesitação."
        )

    # fallback
    if not has_textual_categories and not has_acoustic_flags:
        interpretation_parts.append(
            "Não foram identificados sinais emocionais relevantes na análise textual ou acústica do áudio."
        )

    # conclusão segura
    interpretation_parts.append(
        "Os sinais identificados não representam diagnóstico médico ou psicológico. "
        "A análise de áudio é usada apenas como apoio à triagem preventiva e deve ser interpretada junto com outras evidências."
    )

    return " ".join(interpretation_parts)

def analyze_audio(audio_path: str, language: str = "pt-BR") -> dict:
    audio_file = Path(audio_path)

    if not audio_file.exists():
        raise FileNotFoundError(f"Áudio não encontrado: {audio_path}")

    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(str(audio_file)) as source:
            audio = recognizer.record(source)

        transcription = recognizer.recognize_google(audio, language=language)

    except Exception:
        transcription = "Não foi possível transcrever o áudio."

    voice_features = get_voice_features(str(audio_file))

    emotional_result = classify_emotional_categories(transcription)

    detected_categories = emotional_result["detected_categories"]

    found_flags = emotional_result["flags"] + voice_features.get("tone_flags", [])
    found_flags = list(dict.fromkeys(found_flags))

    risk_score = calculate_audio_risk(
        detected_categories,
        voice_features
    )

    if risk_score >= 0.7:
        risk_level = "alto"
        recommendation = "Recomenda-se avaliação prioritária pela equipe especializada."
    elif risk_score >= 0.4:
        risk_level = "medio"
        recommendation = "Recomenda-se acompanhamento e nova avaliação clínica."
    else:
        risk_level = "baixo"
        recommendation = "Recomenda-se acompanhamento regular."

    interpretation = build_audio_interpretation(
        detected_categories,
        found_flags,
        voice_features
    )

    return {
        "modality": "audio",
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level,
        "flags": found_flags,
        "detected_categories": detected_categories,
        "voice_features": voice_features,
        "transcription": transcription,
        "interpretation": interpretation,
        "recommendation": recommendation
    }