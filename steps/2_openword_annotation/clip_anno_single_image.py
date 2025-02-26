import torch
import clip
from PIL import Image

# 设置多标签阈值（根据场景调整，例如0.2）
threshold = 0.2

# 加载CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)

# 定义候选类别
classes = ["cat", "bird", "turtle", "dog", "anmial"]
text_inputs = clip.tokenize(classes).to(device)


file_path = "/home/tipriest/Pictures/mess11.jpeg"
image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
with torch.no_grad():
    logits_per_image, _ = model(image, text_inputs)
probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

# 提取概率超过阈值的多个类别
selected_classes = [[classes[i], p] for i, p in enumerate(probs)]
for selected_class in selected_classes:
    print(f"the possiblity of {selected_class[0]} is {selected_class[1]}")
