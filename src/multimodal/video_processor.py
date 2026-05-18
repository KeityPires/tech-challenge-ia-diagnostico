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

def analyze_body_posture(frame, pose_model) -> dict:
    posture_flags = []
    posture_interpretation = []

    try:
        results = pose_model(frame, verbose=False)

        frame_height, frame_width = frame.shape[:2]

        for r in results:
            if r.keypoints is None:
                continue

            keypoints = r.keypoints.xy.cpu().numpy()

            for person in keypoints:
                try:
                    left_shoulder = person[5]
                    right_shoulder = person[6]
                    left_hip = person[11]
                    right_hip = person[12]

                    points = [
                        left_shoulder,
                        right_shoulder,
                        left_hip,
                        right_hip
                    ]

                    if any(point[0] <= 0 or point[1] <= 0 for point in points):
                        continue

                    shoulder_width = abs(
                        right_shoulder[0] - left_shoulder[0]
                    )

                    shoulder_height_diff = abs(
                        right_shoulder[1] - left_shoulder[1]
                    )

                    torso_height = abs(
                        ((left_hip[1] + right_hip[1]) / 2)
                        -
                        ((left_shoulder[1] + right_shoulder[1]) / 2)
                    )

                    if shoulder_width <= 0 or torso_height <= 0:
                        continue

                    shoulder_tilt_ratio = shoulder_height_diff / shoulder_width
                    torso_ratio = torso_height / frame_height

                    if shoulder_tilt_ratio > 0.45:
                        posture_flags.append("possible_body_tension")

                    if torso_ratio < 0.14:
                        posture_flags.append("possible_retracted_posture")

                except Exception:
                    continue

    except Exception:
        return {
            "posture_flags": [],
            "posture_score": 0.0,
            "posture_interpretation": []
        }

    posture_flags = list(dict.fromkeys(posture_flags))

    if "possible_body_tension" in posture_flags:
        posture_interpretation.append(
            "Foram observados possíveis sinais posturais de tensão corporal, tratados apenas como evidência complementar e exploratória."
        )

    if "possible_retracted_posture" in posture_flags:
        posture_interpretation.append(
            "Foram observados possíveis sinais de postura retraída, tratados apenas como evidência complementar e exploratória."
        )

    return {
        "posture_flags": posture_flags,
        "posture_score": 0.0,
        "posture_interpretation": posture_interpretation
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
    pose_model = YOLO("yolov8n-pose.pt")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {
            "modality": "video",
            "risk_score": 0,
            "risk_level": "not_provided",
            "flags": ["video_not_opened"],
            "posture_score": 0.0,
            "posture_flags": [],
            "posture_interpretation": [],
            "interpretation": [
                "Não foi possível abrir o vídeo para análise."
            ],
            "limitations": [
                "A análise visual não pôde ser realizada."
            ],
            "metadata": metadata
        }

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    sample_interval_seconds = 0.5
    step = max(1, int(fps * sample_interval_seconds))

    max_frames_to_analyze = 20

    frame_id = 0
    frames_analyzed = 0
    frames_with_faces = 0
    frames_with_pose = 0

    person_detected = False
    total_person_detections = 0
    max_people_in_frame = 0

    faces_detected = 0
    emotions_detected = []
    emotion_confidences = []

    posture_flags_raw = []
    posture_interpretations_raw = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % step == 0:
            if frames_analyzed >= max_frames_to_analyze:
                break

            frames_analyzed += 1
            people_in_frame = 0

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

            posture_result = analyze_body_posture(frame, pose_model)

            if posture_result.get("posture_flags"):
                frames_with_pose += 1
                posture_flags_raw.extend(
                    posture_result.get("posture_flags", [])
                )
                posture_interpretations_raw.extend(
                    posture_result.get("posture_interpretation", [])
                )

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

                                if confidence >= 50:
                                    emotions_detected.append(emotion_type)
                                    emotion_confidences.append(
                                        round(confidence, 2)
                                    )

                                negative_emotions = {
                                    "SAD",
                                    "FEAR",
                                    "ANGRY",
                                    "CONFUSED",
                                    "DISGUSTED"
                                }

                                for emotion in emotions:
                                    emotion_type = emotion["Type"]
                                    confidence = emotion["Confidence"]

                                    if (
                                        emotion_type in negative_emotions
                                        and confidence >= 20
                                    ):
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

    posture_count = Counter(posture_flags_raw)

    posture_flags_detected = []

    if posture_count.get("possible_retracted_posture", 0) >= 4:
        posture_flags_detected.append("possible_retracted_posture")

    if posture_count.get("possible_body_tension", 0) >= 4:
        posture_flags_detected.append("possible_body_tension")

    posture_interpretations = []

    if "possible_body_tension" in posture_flags_detected:
        posture_interpretations.append(
            "Foram observados possíveis sinais posturais persistentes de tensão corporal, tratados apenas como evidência complementar e exploratória."
        )

    if "possible_retracted_posture" in posture_flags_detected:
        posture_interpretations.append(
            "Foram observados possíveis sinais persistentes de postura retraída, tratados apenas como evidência complementar e exploratória."
        )

    posture_score = 0.0

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

    for posture_flag in posture_flags_detected:
        flags.append(posture_flag)

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

    if emotion_percentages:
        emotions_text = ", ".join(
            [
                f"{emotion}: {percentage}%"
                for emotion, percentage in emotion_percentages.items()
            ]
        )

        visual_interpretation.append(
            f"A distribuição percentual das emoções aparentes consideradas na análise foi: {emotions_text}."
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

    if posture_interpretations:
        visual_interpretation.extend(posture_interpretations)

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
        "A análise de postura corporal é complementar, exploratória e não possui valor diagnóstico isolado.",
        "Iluminação, ângulo da câmera, qualidade do vídeo e oclusões podem afetar os resultados.",
        "O YOLOv8 utilizado detecta presença humana e postura corporal aparente, mas não substitui avaliação clínica especializada.",
        "Em vídeos curtos ou bases sintéticas como RAVDESS, a emoção pode variar rapidamente e a análise depende dos frames amostrados."
    ]

    return {
        "modality": "video",
        "risk_score": video_score,
        "risk_level": risk_level,
        "flags": list(dict.fromkeys(flags)),
        "posture_score": posture_score,
        "posture_flags": posture_flags_detected,
        "posture_interpretation": posture_interpretations,
        "interpretation": visual_interpretation,
        "limitations": limitations,
        "metadata": {
            **metadata,
            "frames_analyzed": frames_analyzed,
            "frames_with_faces": frames_with_faces,
            "frames_with_pose": frames_with_pose,
            "faces_detected": faces_detected,
            "person_detected": person_detected,
            "total_person_detections": total_person_detections,
            "max_people_in_frame": max_people_in_frame,
            "dominant_emotions": dict(emotion_count),
            "emotion_percentages": emotion_percentages,
            "emotion_transitions": emotion_transitions,
            "emotion_confidences": emotion_confidences,
            "posture_raw_counts": dict(posture_count),
            "cloud_service": "AWS Rekognition",
            "visual_model": "YOLOv8",
            "pose_model": "YOLOv8 Pose",
            "analysis_strategy": "frame_sampling_every_0_5_seconds_max_20_frames"
        }
    }