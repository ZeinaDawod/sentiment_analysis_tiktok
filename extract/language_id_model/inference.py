
import torch
import librosa
from .model import load_model


def detect_language(audio_path: str) -> str:
    model, feature_extractor = load_model()

    speech_array, _ = librosa.load(audio_path, sr=16000)
    inputs = feature_extractor(speech_array, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_id = torch.argmax(logits, dim=-1).item()
    label = model.config.id2label[predicted_id]
    return label


def is_arabic(audio_path: str) -> bool:

    label = detect_language(audio_path)
    return label.lower().startswith("ar")