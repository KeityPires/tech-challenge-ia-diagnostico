from pathlib import Path
import cv2


def extract_video_metadata(video_path: str) -> dict:
    video_file = Path(video_path)

    if not video_file.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

    cap = cv2.VideoCapture(str(video_file))

    if not cap.isOpened():
        raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_seconds = frame_count / fps if fps else 0

    cap.release()

    return {
        "video_path": str(video_file),
        "fps": round(fps, 2),
        "frame_count": int(frame_count),
        "duration_seconds": round(duration_seconds, 2)
    }


def analyze_video(video_path: str) -> dict:
    metadata = extract_video_metadata(video_path)

    return {
        "modality": "video",
        "risk_score": 0.55,
        "risk_level": "medio",
        "flags": [
            "movimento corporal identificado",
            "análise visual inicial executada"
        ],
        "metadata": metadata
    }