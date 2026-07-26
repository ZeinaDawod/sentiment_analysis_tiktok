import librosa
from .model import load_model


def classify_audio(audio_path: str) -> str:
    classifier = load_model()

    speech_array, sr = librosa.load(audio_path, sr=16000)
    results = classifier({"raw": speech_array, "sampling_rate": sr})

    top_result = results[0]
    return top_result["label"]


def is_music_only(audio_path: str) -> bool:
    label = classify_audio(audio_path).lower()
    return label in ("music", "speech_music")