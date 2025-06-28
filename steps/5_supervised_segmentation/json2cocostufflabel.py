import os
import json
import numpy as np
from PIL import Image
import cv2
from pycocotools import mask as maskUtils
import shutil
import tarfile
import re
import random
import os


def get_filenames_in_folder(folder_path):
    """
    获取指定文件夹下所有文件的文件名（包括后缀）。

    Args:
      folder_path: 文件夹的路径（字符串）。

    Returns:
      一个包含所有文件名的列表（字符串列表）。  如果文件夹不存在或为空，则返回一个空列表。
    """
    try:
        filenames = os.listdir(folder_path)
        return filenames
    except FileNotFoundError:
        print(f"Error: Folder not found at path: {folder_path}")
        return []
    except NotADirectoryError:
        print(f"Error:  {folder_path} is not a directory.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []


def extract_labels(filenames):
    """
    从文件名列表中提取标签。

    Args:
      filenames: 文件名列表（字符串列表）。

    Returns:
      一个包含所有唯一标签的集合（set）。
    """
    labels = set()
    for filename in filenames:
        match = re.search(r"BrushLabels-([^-]+)-0\.png", filename)
        if match:
            label = match.group(1)
            labels.add(label)
    return labels


def generate_distinct_grayscale_values(num_labels, perturbation_range=20):
    """
    生成一组不相似的灰度值。

    Args:
        num_labels: labels 的长度，即需要生成的灰度值数量。
        perturbation_range: 随机扰动的范围，控制灰度值之间的差异程度。

    Returns:
        一个包含灰度值的列表。
    """
    if num_labels <= 0:
        return []

    # 均匀分布的间隔
    interval = 255 // num_labels

    grayscale_values = []
    for i in range(num_labels):
        # 基础灰度值
        base_value = i * interval

        # 添加随机扰动
        perturbation = random.randint(-perturbation_range, perturbation_range)
        grayscale_value = base_value + perturbation

        # 确保灰度值在 0-255 范围内
        grayscale_value = max(0, min(255, grayscale_value))

        grayscale_values.append(int(grayscale_value))  # 转换为整数

    return grayscale_values


def generate_combined_annotations(file_list, label_gray_dict, output_dir):
    """
    为每个 task ID 生成综合的灰度标注图像。

    Args:
        file_list: 文件名列表。
        label_gray_dict: 标签到灰度值的字典。
        output_dir: 输出目录。
    """

    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 提取所有唯一的 task ID
    task_ids = set()
    for filename in file_list:
        task_id = filename.split("-")[1]  # 提取 task ID (例如 "100")
        task_ids.add(task_id)

    for task_id in task_ids:
        # 过滤出当前 task ID 的文件名
        task_files = [f for f in file_list if f.split("-")[1] == task_id]

        # 读取第一张图片以获取尺寸
        first_file = os.path.join(
            "/home/tipriest/Downloads/project-3-at-2025-06-28-04-23-d4d32027",
            task_files[0],
        )
        try:
            img = Image.open(first_file)
            width, height = img.size
            img.close()
        except FileNotFoundError:
            print(f"文件未找到: {first_file}")
            continue
        except Exception as e:
            print(f"打开文件时发生错误: {first_file}, 错误信息: {e}")
            continue

        # 创建一个空白的灰度图像
        combined_annotation = np.zeros((height, width), dtype=np.uint8)

        for filename in task_files:
            try:
                # 提取标签
                label = filename.split("-BrushLabels-")[1].split("-")[
                    0
                ]  # 提取标签 (例如 "concreteroad")

                # 获取对应的灰度值
                gray_value = label_gray_dict.get(label)
                if gray_value is None:
                    print(
                        f"警告: 标签 '{label}' 不在 label_gray_dict 中，跳过文件 {filename}"
                    )
                    continue
                filename = os.path.join(
                    "/home/tipriest/Downloads/project-3-at-2025-06-28-04-23-d4d32027",
                    filename,
                )
                # 读取标注图像
                img = Image.open(filename).convert("L")  # 确保是灰度图像
                annotation_array = np.array(img)
                img.close()

                # 将白色像素 (255) 替换为对应的灰度值
                combined_annotation[annotation_array == 255] = gray_value

            except FileNotFoundError:
                print(f"文件未找到: {filename}")
            except Exception as e:
                print(f"处理文件时发生错误: {filename}, 错误信息: {e}")

        # 保存综合的标注图像
        
        output_filename = f"{int(task_id):04d}.png"
        output_path = os.path.join(output_dir, output_filename)
        try:
            combined_img = Image.fromarray(combined_annotation)
            combined_img.save(output_path)
            combined_img.close()
            print(f"已保存综合标注图像: {output_path}")
        except Exception as e:
            print(f"保存文件时发生错误: {output_path}, 错误信息: {e}")


annotation_folder = (
    "/home/tipriest/Downloads/project-3-at-2025-06-28-04-23-d4d32027"
)
if __name__ == "__main__":
    files = get_filenames_in_folder(annotation_folder)
    labels = extract_labels(files)
    gray_labels = generate_distinct_grayscale_values(len(labels))
    label_gray_dict = dict(zip(labels, gray_labels))
    print(label_gray_dict)
    generate_combined_annotations(
        file_list=files,
        label_gray_dict=label_gray_dict,
        output_dir="/home/tipriest/Documents/30Days-for-segmentation/steps/5_supervised_segmentation/labels",
    )
