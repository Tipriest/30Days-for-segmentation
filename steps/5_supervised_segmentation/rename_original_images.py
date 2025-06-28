"""
Label Studio标注转COCO Stuff格式语义分割标注脚本
功能：根据7个自定义类别生成灰度PNG标注文件
"""

import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

base_dir = "/home/tipriest/Documents/30Days-for-segmentation/steps/5_supervised_segmentation"
input_json = os.path.join(base_dir, "project-3-at-2025-06-28-02-54-6821e755.json")
if __name__ == "__main__":
    with open(input_json) as f:
        data = json.load(f)
    for file in data:
        new_name = file['id']
        old_name = file['file_upload']
        old_name = os.path.join(base_dir, "images/", old_name)
        new_name = os.path.join(base_dir, "images/", f"{int(new_name):04d}.jpg")
        os.rename(old_name, new_name)
