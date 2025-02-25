import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np


import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms

from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from sklearn.metrics import precision_score, recall_score, f1_score
import wandb

# 自定义数据集类
class MultiLabelDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.mlb = MultiLabelBinarizer(classes=classes)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(
            os.getcwd(),
            "./steps/1_preprocess/key_frames",
            self.df.iloc[idx]["filename"],
        )

        labels = self.df.iloc[idx]["labels"].split(",")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label_vec = torch.FloatTensor(self.mlb.fit_transform([labels])[0])
        return image, label_vec


# CAM生成函数（支持多标签选择）
def generate_cam(image_path, target_classes=None):
    file_path = os.path.join(
        os.getcwd(), "steps/1_preprocess/key_frames", image_path
    )
    # 预处理
    image = Image.open(file_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # 创建CAM提取器（必须在forward之前）
    cam_extractor = SmoothGradCAMpp(model, target_layer="layer4")

    # 模型推理
    model.eval()
    with torch.set_grad_enabled(True):  # 修改这里
        # 需要梯度信息
        input_tensor.requires_grad_(True)
        output = model(input_tensor)

    # 获取预测结果
    probs = torch.sigmoid(output).squeeze().cpu().detach().numpy()
    predictions = {classes[i]: float(probs[i]) for i in range(len(classes))}

    # 确定目标类别（如果没有指定则选择置信度最高的三个）
    if not target_classes:
        sorted_indices = np.argsort(probs)[::-1][:3]
        target_classes = [classes[i] for i in sorted_indices]

    # 为每个目标类别生成CAM
    activations = []
    for class_name in target_classes:
        class_idx = classes.index(class_name)
        # 获取对应类别的激活图
        activation = cam_extractor(class_idx, output)
        activations.append(activation[0].squeeze().cpu().numpy())

    # 合并多个激活图（取平均）
    combined_heatmap = np.mean(activations, axis=0)
    combined_heatmap = (combined_heatmap - combined_heatmap.min()) / (
        combined_heatmap.max() - combined_heatmap.min()
    )

    use_visualize = False

    if use_visualize:
        # 可视化
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))

        # 原始图像
        ax[0].imshow(image)
        ax[0].set_title("Original Image")
        ax[0].axis("off")

        # 热力图
        ax[1].imshow(combined_heatmap, cmap="jet")
        ax[1].set_title("Class Activation Map")
        ax[1].axis("off")

        # 叠加结果
        result = overlay_mask(
            image, Image.fromarray(combined_heatmap), alpha=0.5
        )
        ax[2].imshow(result)
        ax[2].set_title("Overlay Visualization")
        ax[2].axis("off")

        plt.suptitle(
            f"Predictions: {predictions}\nTarget Classes: {target_classes}"
        )
        plt.tight_layout()
        plt.show()
    save_base_path = os.path.join(os.getcwd(), "steps/3_gen_ini_cam/results")
    result = overlay_mask(image, Image.fromarray(combined_heatmap), alpha=0.5)
    result.save(f"{save_base_path}/{image_path}")

    # 必须清理hook
    cam_extractor.remove_hooks()

if __name__ == "__main__":
    wandb.init(project="pseudo", entity="tipriest-hit")

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取CSV文件并构建类别编码器
    df = pd.read_csv(
        "./steps/2_openword_annotation/labels_groundtruth.csv",
        sep=",",
        header=None,
        names=["filename", "labels"],
    )
    all_labels = [labels.split(",") for labels in df["labels"]]
    mlb = MultiLabelBinarizer()
    encoded_labels = mlb.fit_transform(all_labels)
    classes = mlb.classes_.tolist()  # 获取完整类别列表

    # 数据预处理
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    # 创建数据集和数据加载器
    dataset = MultiLabelDataset(df, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 修改模型为多标签分类
    model = torchvision.models.resnet101(weights="IMAGENET1K_V2")
    model.fc = torch.nn.Linear(2048, len(classes))
    model = model.to(device)

    # 训练时使用BCEWithLogitsLoss
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    wandb.watch(model, criterion, log="all")

    for epoch in range(100):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算预测结果
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            # 收集指标数据
            all_preds.append(preds.cpu().detach())
            all_labels.append(labels.cpu().detach())

            # 计算批次准确率(汉明准确率)
            batch_correct = (preds == labels).sum().item()
            total_correct += batch_correct
            total_samples += (
                labels.numel()
            )  # 总标签数 = batch_size * num_classes

            # 累计损失
            total_loss += loss.item() * inputs.size(0)

        # 计算epoch指标
        epoch_loss = total_loss / total_samples
        epoch_accuracy = total_correct / total_samples

        # 合并所有预测和标签
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # 计算更复杂的指标（确保数据在CPU上）
        epoch_precision = precision_score(all_labels, all_preds, average="micro")
        epoch_recall = recall_score(all_labels, all_preds, average="micro")
        epoch_f1 = f1_score(all_labels, all_preds, average="micro")

        # 记录到wandb
        wandb.log(
            {
                "train_loss": epoch_loss,
                "train_accuracy": epoch_accuracy,
                "train_precision": epoch_precision,
                "train_recall": epoch_recall,
                "train_f1": epoch_f1,
            }
        )

    with open("./steps/2_openword_annotation/labels_groundtruth.csv", "r") as f:
        for line in f:
            parts = [part.strip().replace('"', "") for part in line.split(",")]
            pic_name = parts[0]
            target_classes = parts[1:]
            generate_cam(pic_name, target_classes)
    # generate_cam("000043.jpg", target_classes=["cement road", "red brick road"])
