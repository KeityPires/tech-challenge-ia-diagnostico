from pathlib import Path
import speech_recognition as sr


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

    risk_keywords = ["medo", "cansaço", "dificuldade para dormir", "ansiedade"]

    found_flags = [
        keyword for keyword in risk_keywords
        if keyword.lower() in transcription.lower()
    ]

    risk_score = min(0.3 + 0.15 * len(found_flags), 1.0)

    return {
        "modality": "audio",
        "risk_score": round(risk_score, 2),
        "risk_level": "alto" if risk_score >= 0.7 else "medio",
        "flags": found_flags,
        "transcription": transcription
    }