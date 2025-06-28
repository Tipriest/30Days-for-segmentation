import os
import random
import shutil


def split_data(images_dir, labels_dir, train_ratio=0.8):
    """
    将图像和标注数据按照指定的比例随机分割成训练集和验证集。

    Args:
        images_dir (str): 图像文件夹的路径。
        labels_dir (str): 标注文件夹的路径。
        train_ratio (float): 训练集比例，范围为 (0, 1)。
    """

    # 1. 创建 train 和 val 文件夹
    train_images_dir = os.path.join(images_dir, "train")
    val_images_dir = os.path.join(images_dir, "val")
    train_labels_dir = os.path.join(labels_dir, "train")
    val_labels_dir = os.path.join(labels_dir, "val")

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # 2. 获取所有图像文件名 (假设图像和标注文件名相同)
    image_files = [
        f
        for f in os.listdir(images_dir)
        if os.path.isfile(os.path.join(images_dir, f))
        and not f.startswith(".")
        and f.lower().endswith(".jpg")  # 过滤掉隐藏文件和非jpg文件
    ]  # 过滤掉隐藏文件
    image_files = [
        f for f in image_files if f not in ["train", "val"]
    ]  # 过滤掉train和val文件夹

    # 3. 随机打乱文件名列表
    random.shuffle(image_files)

    # 4. 计算训练集和验证集的大小
    num_images = len(image_files)
    num_train = int(num_images * train_ratio)

    # 5. 分割文件名列表
    train_files = image_files[:num_train]
    val_files = image_files[num_train:]

    # 6. 移动文件到对应的文件夹
    def move_files(
        file_list,
        src_images_dir,
        src_labels_dir,
        dest_images_dir,
        dest_labels_dir,
    ):
        for filename in file_list:
            # 移动图像文件
            src_image_path = os.path.join(src_images_dir, filename)
            dest_image_path = os.path.join(dest_images_dir, filename)
            shutil.move(src_image_path, dest_image_path)

            # 移动标注文件 (假设标注文件名与图像文件名相同，只是扩展名不同)
            name, ext = os.path.splitext(filename)
            label_filename = (
                name + ".png"
            )  # 假设标注文件是 .png 格式，根据实际情况修改
            src_label_path = os.path.join(labels_dir, label_filename)
            dest_label_path = os.path.join(dest_labels_dir, label_filename)

            # 检查标注文件是否存在，如果不存在则跳过
            if os.path.exists(src_label_path):
                shutil.move(src_label_path, dest_label_path)
            else:
                print(
                    f"Warning: Label file not found for {filename}. Skipping."
                )

    move_files(
        train_files, images_dir, labels_dir, train_images_dir, train_labels_dir
    )
    move_files(
        val_files, images_dir, labels_dir, val_images_dir, val_labels_dir
    )

    print(f"Successfully split data into train and val sets.")
    print(f"Train images: {len(train_files)}")
    print(f"Val images: {len(val_files)}")


# 示例用法
if __name__ == "__main__":
    base_dir = os.getcwd()
    images_dir = os.path.join(base_dir, "images")  # 替换为你的图像文件夹路径
    labels_dir = os.path.join(base_dir, "labels")  # 替换为你的标注文件夹路径
    train_ratio = 0.75  # 训练集比例

    split_data(images_dir, labels_dir, train_ratio)
