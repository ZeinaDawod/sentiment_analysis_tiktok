

try:
    from speechbrain.inference.interfaces import foreign_class
except ImportError:

    from speechbrain.pretrained import foreign_class

MODEL_SOURCE = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
SAVE_DIR = "pretrained_models/tone_model"

_classifier = None


def load_model():
    global _classifier
    if _classifier is None:
        print(f"[tone_model] Loading model {MODEL_SOURCE} ...")
        _classifier = foreign_class(
            source=MODEL_SOURCE,
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            savedir=SAVE_DIR,
        )
    return _classifier