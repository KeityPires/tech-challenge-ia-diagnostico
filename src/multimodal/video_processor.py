from pathlib import Path
from collections import Counter
import os

import cv2
import boto3

from dotenv import load_dotenv
from ultralytics import YOLO


def extract_video_metadata(video_path: str) -> dict:
    video_file = Path(video_path)

    if not video_file.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

    cap = cv2.VideoCapture(str(video_file))

    if not cap.isOpened():
        return {
            "video_path": str(video_file),
            "fps": 0,
            "frame_count": 0,
            "duration_seconds": 0,
            "status": "mock_video_file"
        }

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
    load_dotenv()

    metadata = extract_video_metadata(video_path)

    region = os.getenv("AWS_REGION", "us-east-1")

    rekognition = boto3.client(
        "rekognition",
        region_name=region
    )

    yolo_model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {
            "modality": "video",
            "risk_score": 0,
            "risk_level": "not_provided",
            "flags": ["video_not_opened"],
            "interpretation": [
                "Não foi possível abrir o vídeo para análise."
            ],
            "limitations": [
                "A análise visual não pôde ser realizada."
            ],
            "metadata": metadata
        }

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    # Para vídeos curtos, como RAVDESS, analisamos frames com maior frequência.
    sample_interval_seconds = 0.5
    step = max(1, int(fps * sample_interval_seconds))

    max_frames_to_analyze = 20

    frame_id = 0
    frames_analyzed = 0
    frames_with_faces = 0

    person_detected = False
    total_person_detections = 0
    max_people_in_frame = 0

    faces_detected = 0
    emotions_detected = []
    emotion_confidences = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % step == 0:
            if frames_analyzed >= max_frames_to_analyze:
                break

            frames_analyzed += 1
            people_in_frame = 0

            # YOLOv8: detecção de presença humana.
            yolo_results = yolo_model(frame, verbose=False)

            for r in yolo_results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = yolo_model.names[cls]

                    if label == "person":
                        person_detected = True
                        people_in_frame += 1
                        total_person_detections += 1

            max_people_in_frame = max(
                max_people_in_frame,
                people_in_frame
            )

            # AWS Rekognition: emoções faciais aparentes.
            success, encoded_image = cv2.imencode(".jpg", frame)

            if success:
                try:
                    response = rekognition.detect_faces(
                        Image={"Bytes": encoded_image.tobytes()},
                        Attributes=["ALL"]
                    )

                    faces = response.get("FaceDetails", [])

                    if faces:
                        frames_with_faces += 1
                        faces_detected += len(faces)

                        for face in faces:
                            emotions = face.get("Emotions", [])

                            if emotions:
                                dominant_emotion = max(
                                    emotions,
                                    key=lambda e: e["Confidence"]
                                )

                                emotion_type = dominant_emotion["Type"]
                                confidence = dominant_emotion["Confidence"]

                                # Limiar menor para datasets curtos/sintéticos,
                                # como RAVDESS.
                                if confidence >= 50:
                                    emotions_detected.append(emotion_type)
                                    emotion_confidences.append(
                                        round(confidence, 2)
                                    )

                except Exception:
                    emotions_detected.append("UNKNOWN")
                    emotion_confidences.append(0)

        frame_id += 1

    cap.release()

    emotion_count = Counter(emotions_detected)
    total_emotions = sum(emotion_count.values())

    if total_emotions > 0:
        emotion_percentages = {
            emotion: round((count / total_emotions) * 100, 2)
            for emotion, count in emotion_count.items()
        }
    else:
        emotion_percentages = {}

    emotion_transitions = 0

    for i in range(1, len(emotions_detected)):
        if emotions_detected[i] != emotions_detected[i - 1]:
            emotion_transitions += 1

    emotion_weights = {
        "FEAR": 0.8,
        "SAD": 0.6,
        "ANGRY": 0.5,
        "DISGUSTED": 0.4,
        "SURPRISED": 0.3,
        "CONFUSED": 0.3,
        "CALM": 0.1,
        "UNKNOWN": 0.2,
        "HAPPY": 0.0
    }

    if emotions_detected:
        scores = [
            emotion_weights.get(emotion, 0.2)
            for emotion in emotions_detected
        ]
        video_score = sum(scores) / len(scores)
    else:
        video_score = 0

    flags = []

    if person_detected:
        flags.append("person_detected")

    if faces_detected > 0:
        flags.append("face_detected")

    if person_detected and faces_detected == 0:
        flags.append("face_not_visible")

    if max_people_in_frame >= 3:
        flags.append("multiple_people_detected")

    for emotion in [
        "FEAR",
        "SAD",
        "ANGRY",
        "DISGUSTED",
        "CONFUSED"
    ]:
        if emotion_count.get(emotion, 0) > 0:
            flags.append(f"{emotion.lower()}_expression")

    if emotion_percentages.get("FEAR", 0) >= 20:
        flags.append("persistent_fear")
        video_score += 0.1

    if emotion_percentages.get("SAD", 0) >= 20:
        flags.append("persistent_sadness")
        video_score += 0.1

    if emotion_percentages.get("ANGRY", 0) >= 15:
        flags.append("persistent_tension")
        video_score += 0.1

    if emotion_percentages.get("CONFUSED", 0) >= 20:
        flags.append("persistent_confusion")
        video_score += 0.05

    if emotion_transitions >= 5:
        flags.append("emotional_variation_detected")
        video_score += 0.05

    # Se não houve nenhuma face detectada, não atribuímos risco emocional visual.
    if faces_detected == 0:
        video_score = 0

    video_score = round(min(video_score, 1.0), 2)

    if video_score >= 0.75:
        risk_level = "alto"
    elif video_score >= 0.45:
        risk_level = "medio"
    elif video_score > 0:
        risk_level = "baixo"
    else:
        risk_level = "not_provided"

    visual_interpretation = []

    if not person_detected:
        visual_interpretation.append(
            "Não foi detectada presença humana de forma confiável no vídeo."
        )

    if person_detected and faces_detected == 0:
        visual_interpretation.append(
            "Foi detectada presença humana, mas não houve detecção facial suficiente para análise emocional."
        )

    if faces_detected > 0:
        visual_interpretation.append(
            "Foram detectadas faces no vídeo, permitindo análise de expressões emocionais aparentes."
        )

    if "persistent_fear" in flags:
        visual_interpretation.append(
            "Foram observados sinais visuais persistentes associados a medo aparente."
        )

    if "persistent_sadness" in flags:
        visual_interpretation.append(
            "Foram observados sinais visuais persistentes associados a tristeza aparente."
        )

    if "persistent_tension" in flags:
        visual_interpretation.append(
            "Foram observados sinais visuais persistentes associados a tensão ou raiva aparente."
        )

    if "persistent_confusion" in flags:
        visual_interpretation.append(
            "Foram observados sinais visuais persistentes associados a confusão ou insegurança aparente."
        )

    if "emotional_variation_detected" in flags:
        visual_interpretation.append(
            "Foram identificadas variações emocionais ao longo do vídeo, usadas como evidência complementar."
        )

    if max_people_in_frame >= 3:
        visual_interpretation.append(
            "O vídeo apresentou múltiplas pessoas no mesmo frame, indicando maior complexidade contextual da cena."
        )

    if faces_detected > 0 and not any(
        flag in flags
        for flag in [
            "fear_expression",
            "sad_expression",
            "angry_expression",
            "disgusted_expression",
            "confused_expression",
            "persistent_fear",
            "persistent_sadness",
            "persistent_tension",
            "persistent_confusion"
        ]
    ):
        visual_interpretation.append(
            "A análise facial não identificou sinais visuais relevantes de desconforto emocional."
        )

    limitations = [
        "A análise facial indica apenas expressões aparentes, não estado psicológico real.",
        "O sistema não realiza diagnóstico médico ou psicológico.",
        "Os sinais visuais devem ser interpretados apenas como apoio à triagem.",
        "Iluminação, ângulo da câmera, qualidade do vídeo e oclusões podem afetar os resultados.",
        "O YOLOv8 utilizado detecta presença humana, mas não substitui avaliação clínica especializada.",
        "Em vídeos curtos ou bases sintéticas como RAVDESS, a emoção pode variar rapidamente e a análise depende dos frames amostrados."
    ]

    return {
        "modality": "video",
        "risk_score": video_score,
        "risk_level": risk_level,
        "flags": list(dict.fromkeys(flags)),
        "interpretation": visual_interpretation,
        "limitations": limitations,
        "metadata": {
            **metadata,
            "frames_analyzed": frames_analyzed,
            "frames_with_faces": frames_with_faces,
            "faces_detected": faces_detected,
            "person_detected": person_detected,
            "total_person_detections": total_person_detections,
            "max_people_in_frame": max_people_in_frame,
            "dominant_emotions": dict(emotion_count),
            "emotion_percentages": emotion_percentages,
            "emotion_transitions": emotion_transitions,
            "emotion_confidences": emotion_confidences,
            "cloud_service": "AWS Rekognition",
            "visual_model": "YOLOv8",
            "analysis_strategy": "frame_sampling_every_0_5_seconds_max_20_frames"
        }
    }