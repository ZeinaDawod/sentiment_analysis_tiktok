import re
import torch
from .model import load_model


def _split_sentences(text: str) -> list:
    
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]


def _correct_single_sentence(sentence: str, model, tokenizer) -> str:
    input_text = "correct grammar: " + sentence
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True).input_ids

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=128, num_beams=4)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def correct_spelling(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text

    model, tokenizer = load_model()

    sentences = _split_sentences(text)
    corrected_sentences = [_correct_single_sentence(s, model, tokenizer) for s in sentences]

    return " ".join(corrected_sentences)