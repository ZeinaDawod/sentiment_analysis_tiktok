from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
from config import LANGUAGE_ID_MODEL
import logging
MODEL_NAME = LANGUAGE_ID_MODEL

_model = None
_feature_extractor = None
logger = logging.getLogger("language_id_model")


def load_model():
    global _model, _feature_extractor
    if _model is None:
        logger.info(f"[language_id_model] Loading model {MODEL_NAME} ...")
        _feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        _model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()
    return _model, _feature_extractor
