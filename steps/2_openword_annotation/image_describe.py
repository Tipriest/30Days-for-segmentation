from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
import spacy

# 加载BLIP模型
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to("cuda")

# 生成图像描述
image = Image.open("frame.jpg").convert("RGB")
inputs = processor(image, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_length=50)
caption = processor.decode(
    out[0], skip_special_tokens=True
)  # e.g., "a road next to a grassy field with water"



nlp = spacy.load("en_core_web_sm")
doc = nlp(caption)
candidate_classes = [
    chunk.text for chunk in doc.noun_chunks
]  # e.g., ["road", "grassy field", "water"]
