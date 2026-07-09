from .model import load_model

def predict_tone(audio_path: str) -> str:
    classifier = load_model()

    result = classifier(audio_path)

    return result[0]["label"]