import torch
from PIL import Image
from collections import Counter
from .model import load_model
from ..bais import extract_frames
from config import NUM_FRAMES

NUM_FRAMES=int(NUM_FRAMES)
def predict_facial_expression(video_path: str, num_frames: int = NUM_FRAMES) -> str:
    model, processor = load_model()

    frame_paths = extract_frames(video_path, num_frames=num_frames)

    labels = []
    for frame_path in frame_paths:
        image = Image.open(frame_path).convert("RGB")
        inputs = processor(image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        predicted_id = outputs.logits.argmax(-1).item()
        label = model.config.id2label[predicted_id]
        labels.append(label)


    most_common_label = Counter(labels).most_common(1)[0][0]
    return most_common_label