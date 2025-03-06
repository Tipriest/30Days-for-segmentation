import os
import csv
from runpy import run_path
import torch
import clip
from PIL import Image
import utils


# 设置多标签阈值（根据场景调整，例如0.2）
threshold = 0.2

# 加载CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)

# 定义候选类别
classes = [
    "cement road",
    "red paved path",
    "yellow paved path",
    "soil",
    "lawn",
    "water",
    "curb",
    "others",
]
text_inputs = clip.tokenize(classes).to(device)

# 创建CSV文件保存标签
CSV_FILE_PATH = os.path.join(
    os.getcwd(), "./steps/2_openword_annotation/anno_results/labels.csv"
)
KEY_FRAMES_PATH = os.path.join(os.getcwd(), "./steps/1_preprocess/key_frames")
with utils.timer():
    with open(CSV_FILE_PATH, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Frame", "Labels"])  # 表头

        for frame_path in os.listdir(KEY_FRAMES_PATH):
            file_path = os.path.join(KEY_FRAMES_PATH, frame_path)
            image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                logits_per_image, _ = model(image, text_inputs)
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

            # 提取概率超过阈值的多个类别
            selected_classes = [
                classes[i] for i, p in enumerate(probs) if p > threshold
            ]

            # 如果没有超过阈值的类，保留最高概率的类（避免空标签）
            if not selected_classes:
                selected_classes = [classes[probs.argmax()]]

            # 写入CSV（例如：frame0001.jpg -> 草地,水面）
            writer.writerow([frame_path, ",".join(selected_classes)])

    run_path("./steps/2_openword_annotation/anno_results/calF1.py")
