import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
from config import QWEN_MODEL

logger = logging.getLogger("sentiment_analysis")

MODEL_NAME = QWEN_MODEL

_generator = None


def load_model():
    global _generator
    if _generator is None:
        logger.info(f"Loading model {MODEL_NAME} (8-bit quantized) ...")

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        _generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        logger.info("Model loaded successfully")

    return _generator