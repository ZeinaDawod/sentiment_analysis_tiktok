from transformers import pipeline
from config import AUDIO_MODEL
MODEL_NAME = AUDIO_MODEL

_classifier = None


def load_model():
    global _classifier
    if _classifier is None:
        print(f"[audio_classifier] Loading model {MODEL_NAME} ...")
        _classifier = pipeline("audio-classification", model=MODEL_NAME)
    return _classifier