from pathlib import Path
from moviepy import VideoFileClip


def extract_audio_from_video(
    video_path: str,
    output_audio_path: str | None = None,
    audio_format: str = "wav"
) -> str:
    video_file = Path(video_path)

    if not video_file.exists():
        raise FileNotFoundError(
            f"Vídeo não encontrado: {video_path}"
        )

    if output_audio_path is None:
        output_audio_path = str(
            video_file.with_suffix(f".{audio_format}")
        )

    output_audio_file = Path(output_audio_path)
    output_audio_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    video = VideoFileClip(str(video_file))

    if video.audio is None:
        video.close()
        raise ValueError(
            "O vídeo não possui faixa de áudio."
        )

    video.audio.write_audiofile(
        str(output_audio_file),
        logger=None
    )

    video.close()

    return str(output_audio_file)