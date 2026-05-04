from pathlib import Path
import wave
import audioop
import speech_recognition as sr


def get_voice_features(audio_path: str) -> dict:
    audio_file = Path(audio_path)

    try:
        with wave.open(str(audio_file), "rb") as wav:
            frames = wav.readframes(wav.getnframes())
            sample_width = wav.getsampwidth()
            frame_rate = wav.getframerate()
            frame_count = wav.getnframes()
            duration_seconds = frame_count / frame_rate if frame_rate else 0
            rms = audioop.rms(frames, sample_width)

        if rms > 1800:
            voice_intensity = "alta"
        elif rms > 800:
            voice_intensity = "moderada"
        else:
            voice_intensity = "baixa"

        return {
            "duration_seconds": round(duration_seconds, 2),
            "rms_energy": rms,
            "voice_intensity": voice_intensity
        }

    except Exception:
        return {
            "duration_seconds": None,
            "rms_energy": None,
            "voice_intensity": "indisponível"
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
            "ela não deixa", "tenho medo", "melhor eu ir", "não quero responder"
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
    score = 0.25 + (0.15 * category_count)

    if "sinais_de_violencia_ou_medo" in detected_categories:
        score += 0.2

    if "depressao_pos_parto_ou_sofrimento_emocional" in detected_categories:
        score += 0.2

    if voice_features.get("voice_intensity") == "alta":
        score += 0.1

    return min(score, 1.0)


def build_audio_interpretation(detected_categories: dict, flags: list) -> str:
    if not detected_categories:
        return (
            "Não foram identificados sinais emocionais relevantes na transcrição. "
            "A recomendação é manter acompanhamento regular."
        )

    categories_text = ", ".join(detected_categories.keys())
    flags_text = ", ".join(flags)

    return (
        f"A análise identificou categorias associadas a {categories_text}. "
        f"Os principais indicadores encontrados foram: {flags_text}. "
        "Esses sinais não representam diagnóstico, mas podem indicar necessidade de atenção especializada."
    )


def analyze_audio(audio_path: str) -> dict:
    audio_file = Path(audio_path)

    if not audio_file.exists():
        raise FileNotFoundError(f"Áudio não encontrado: {audio_path}")

    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(str(audio_file)) as source:
            audio = recognizer.record(source)

        transcription = recognizer.recognize_google(audio, language="pt-BR")

    except Exception:
        transcription = "Não foi possível transcrever o áudio."

    voice_features = get_voice_features(str(audio_file))
    emotional_result = classify_emotional_categories(transcription)

    detected_categories = emotional_result["detected_categories"]
    found_flags = emotional_result["flags"]

    risk_score = calculate_audio_risk(detected_categories, voice_features)

    if risk_score >= 0.7:
        risk_level = "alto"
        recommendation = "Recomenda-se avaliação prioritária pela equipe especializada."
    elif risk_score >= 0.4:
        risk_level = "medio"
        recommendation = "Recomenda-se acompanhamento e nova avaliação clínica."
    else:
        risk_level = "baixo"
        recommendation = "Recomenda-se acompanhamento regular."

    interpretation = build_audio_interpretation(detected_categories, found_flags)

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