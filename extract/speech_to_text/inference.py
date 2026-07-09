from .model import load_model

def transcribe(audio_path: str, language: str = None) -> str:
    asr_pipeline = load_model()

    if language is None:
        result = asr_pipeline(audio_path)
    else:
        result = asr_pipeline(
            audio_path,
            generate_kwargs={
                "language": language,
                "task": "transcribe",
            },
        )

    return result["text"].strip()