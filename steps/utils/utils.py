import csv
import time
from contextlib import contextmanager

# from pathlib import Path
import os
import base64
import logging
from io import BytesIO
from PIL import Image


def convert_to_base64(_pil_image: Image):
    _buffered = BytesIO()
    if _pil_image.mode == "RGBA":
        _pil_image = _pil_image.convert("RGB")
    _pil_image.save(_buffered, format="JPEG")
    img_str = base64.b64encode(_buffered.getvalue()).decode("utf-8")
    return img_str


def load_image_labels(csv_path, base_dir=""):
    """
    从CSV加载图像路径和对应的多标签
    参数：
        csv_path: CSV文件路径
        base_dir: 图像路径前缀（若CSV中的路径为相对路径）
    """
    image_labels = {}
    with open(csv_path, "r", encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 处理图像路径
            img_rel_path = row["image_path"]
            img_full_path = os.path.join(base_dir, img_rel_path)
            # 处理标签（假设labels列为逗号分隔）
            labels = [lbl.strip().lower() for lbl in row["labels"].split(",")]
            image_labels[img_full_path] = labels
    return image_labels


# # 示例用法
# csv_path = "your_data/labels.csv"
# image_labels = load_image_labels(csv_path, base_dir="your_data/")


def save_to_csv(_results, output_path="labels_llm_pred.csv"):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Labels"])
        for _result in _results:
            # writer.writerow(
            #     [
            #         result["image_path"],
            #         result["annotations"],
            #     ]
            # )
            # 写入CSV（例如：frame0001.jpg -> 草地,水面）
            writer.writerow(
                [_result["image_path"], ",".join(_result["annotations"])]
            )


@contextmanager
def timer():
    start = time.perf_counter()
    yield  # 在此处插入被计时代码
    end = time.perf_counter()
    print(f"运行时长: {end - start:.6f} 秒")


def send_alert(message):
    logging.error("ALERT: %s", message)
    # 可扩展：同时写入文件/发送到日志服务器
