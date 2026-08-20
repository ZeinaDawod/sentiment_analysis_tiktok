from transformers import pipeline
from config import WHISPER_MODEL
import logging
MODEL_NAME =  WHISPER_MODEL

_asr_pipeline = None
logger = logging.getLogger("speech_to_text")


def load_model():
    global _asr_pipeline
    if _asr_pipeline is None:
        logger.info(f"[speech_to_text] Loading model {MODEL_NAME} ...")
        _asr_pipeline = pipeline("automatic-speech-recognition", model=MODEL_NAME)
    return _asr_pipeline