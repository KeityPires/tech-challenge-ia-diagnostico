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

    # cliente AWS Rekognition

    region = os.getenv("AWS_REGION", "us-east-1")

    rekognition = boto3.client(
        "rekognition",
        region_name=region
    )

    # modelo YOLOv8

    yolo_model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {
            "modality": "video",
            "risk_score": 0,
            "risk_level": "not_provided",
            "flags": ["video_not_opened"],
            "metadata": metadata
        }

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    # 1 frame a cada 5 segundos

    step = fps * 5

    frame_id = 0
    frames_analyzed = 0

    person_detected = False
    faces_detected = 0

    emotions_detected = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % step == 0:

            frames_analyzed += 1

            # YOLOv8 → detecção de pessoa

            yolo_results = yolo_model(frame, verbose=False)

            for r in yolo_results:
                for box in r.boxes:

                    cls = int(box.cls[0])
                    label = yolo_model.names[cls]

                    if label == "person":
                        person_detected = True

            
            # AWS Rekognition → emoções faciais            

            success, encoded_image = cv2.imencode(".jpg", frame)

            if success:

                try:
                    response = rekognition.detect_faces(
                        Image={"Bytes": encoded_image.tobytes()},
                        Attributes=["ALL"]
                    )

                    faces = response.get("FaceDetails", [])

                    if faces:

                        faces_detected += len(faces)

                        for face in faces:

                            emotions = face.get("Emotions", [])

                            if emotions:

                                dominant_emotion = max(
                                    emotions,
                                    key=lambda e: e["Confidence"]
                                )

                                emotions_detected.append(
                                    dominant_emotion["Type"]
                                )

                except Exception:
                    emotions_detected.append("UNKNOWN")

        frame_id += 1

    cap.release()

    # cálculo de score emocional
    
    emotion_weights = {
        "FEAR": 0.8,
        "SAD": 0.6,
        "ANGRY": 0.5,
        "DISGUSTED": 0.4,
        "SURPRISED": 0.3,
        "CONFUSED": 0.3,
        "CALM": 0.2,
        "UNKNOWN": 0.2,
        "HAPPY": 0.0
    }

    if emotions_detected:

        scores = [
            emotion_weights.get(emotion, 0.2)
            for emotion in emotions_detected
        ]

        video_score = round(sum(scores) / len(scores), 2)

    else:
        video_score = 0
    
    # classificação de risco    

    if video_score >= 0.75:
        risk_level = "alto"

    elif video_score >= 0.45:
        risk_level = "medio"

    elif video_score > 0:
        risk_level = "baixo"

    else:
        risk_level = "not_provided"
    
    # flags    

    flags = []

    if person_detected:
        flags.append("person_detected")

    if faces_detected > 0:
        flags.append("face_detected")

    emotion_count = Counter(emotions_detected)

    for emotion in [
        "FEAR",
        "SAD",
        "ANGRY",
        "DISGUSTED",
        "CONFUSED"
    ]:

        if emotion_count.get(emotion, 0) > 0:
            flags.append(f"{emotion.lower()}_expression")
    
    # retorno final    

    return {
        "modality": "video",
        "risk_score": video_score,
        "risk_level": risk_level,
        "flags": flags,
        "metadata": {
            **metadata,
            "frames_analyzed": frames_analyzed,
            "faces_detected": faces_detected,
            "dominant_emotions": dict(emotion_count),
            "cloud_service": "AWS Rekognition",
            "visual_model": "YOLOv8"
        }
    }