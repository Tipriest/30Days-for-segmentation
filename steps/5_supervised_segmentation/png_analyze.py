from PIL import Image
import numpy as np


def analyze_png_with_pillow(filepath):
    """使用Pillow库解析PNG图片"""
    try:
        with Image.open(filepath) as img:
            print(f"\n分析文件: {filepath}")
            print(f"格式: {img.format}")
            print(f"模式: {img.mode}")  # 如RGB, RGBA, L(灰度)等
            print(f"尺寸: {img.size} (宽x高)")
            print(f"位深度: {img.bits} bits/通道")

            # 转换为numpy数组获取更多信息
            img_array = np.array(img)
            print(
                f"实际通道数: {img_array.shape[-1] if len(img_array.shape) == 3 else 1}"
            )
            print(f"数据类型: {img_array.dtype}")

    except Exception as e:
        print(f"错误: {e}")


# 使用示例
analyze_png_with_pillow("/data/cocostuff/annotations/000000005193.png")
