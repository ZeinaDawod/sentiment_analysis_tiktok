from transformers import pipeline

MODEL_NAME = "openai/whisper-small"

_asr_pipeline = None


def load_model():
    global _asr_pipeline
    if _asr_pipeline is None:
        print(f"[speech_to_text] Loading model {MODEL_NAME} ...")
        _asr_pipeline = pipeline("automatic-speech-recognition", model=MODEL_NAME)
    return _asr_pipeline