from transformers import T5Tokenizer, T5ForConditionalGeneration
from config import SPELLING_CORRECT
import logging
MODEL_NAME = SPELLING_CORRECT

_model = None
_tokenizer = None
logger=logging.getLogger("spelling_correction_model")


def load_model():
    global _model, _tokenizer
    if _model is None:
        logger.info(f"[spelling_correction_model] Loading model {MODEL_NAME} ...")
        _tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        _model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        _model.eval()
    return _model, _tokenizer
