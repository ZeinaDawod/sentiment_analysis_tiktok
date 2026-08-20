
from transformers import ViTImageProcessor, AutoModelForImageClassification

import logging
from config import FACE_MODEL

MODEL_NAME = FACE_MODEL

_model = None
_processor = None

logger = logging.getLogger("facial_expression_model")


def load_model():
    global _model, _processor
    if _model is None:
        logger.info(f"[facial_expression_model] Loading model {MODEL_NAME} ...")
        _processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
        _model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        _model.eval()
    return _model, _processor