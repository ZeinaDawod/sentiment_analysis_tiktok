import json
import logging
from .model import load_model

logger = logging.getLogger("sentiment_analysis")

_decoder = json.JSONDecoder()


def extract_first_json(text: str):
    text = text.strip()
    start = text.find("{")
    if start == -1:
        return None
    try:
        obj, _ = _decoder.raw_decode(text[start:])
        return obj
    except json.JSONDecodeError:
        return None


def generate_sentiment(transcript, description, expression, prompt_template):
    generator = load_model()

    prompt = prompt_template.format(
        transcript=transcript, description=description, expression=expression
    )

    result = generator(prompt, max_new_tokens=200, do_sample=False, return_full_text=False)
    raw_output = result[0]["generated_text"].strip()

    parsed = extract_first_json(raw_output)

    if parsed is None:
        logger.debug(f"Failed to parse JSON from: {raw_output[:100]}")
        return {"raw_output": raw_output, "predicted_sentiment": "unknown"}

    model_stated_class = parsed.get("class", "unknown")
    if model_stated_class not in ("positive", "negative", "neutral"):
        model_stated_class = "unknown"

    return {"raw_output": raw_output, "predicted_sentiment": model_stated_class}