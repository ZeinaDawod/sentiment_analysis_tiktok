from .model import load_model


def transcribe(audio_path: str) -> str:
    asr_pipeline = load_model()

    result = asr_pipeline(
        audio_path,
        return_timestamps=True,
        generate_kwargs={
            "language": "english",
            "task": "transcribe",
        },
    )

    return result["text"].strip()