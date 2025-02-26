import pandas as pd

# 读取数据（假设列名为Frame和Labels）
true_df = pd.read_csv(
    "./steps/2_openword_annotation/labels_groundtruth.csv"
)  # 真实标签
pred_df = pd.read_csv("./steps/2_openword_annotation/labels.csv")  # 预测标签

# 合并数据并处理标签格式
merged = pd.merge(
    true_df.rename(columns={"Labels": "Labels_true"}),
    pred_df.rename(columns={"Labels": "Labels_pred"}),
    on="Frame",
)


# 清洗标签格式
def clean_labels(s):
    return {item.strip() for item in s.split(",") if item.strip()}


merged["labels_true"] = merged["Labels_true"].apply(clean_labels)
merged["labels_pred"] = merged["Labels_pred"].apply(clean_labels)

# 获取所有唯一标签
all_labels = set()
merged["labels_true"].apply(all_labels.update)
merged["labels_pred"].apply(all_labels.update)
all_labels = list(all_labels)

# ---------------------- 新增：统计真实标注频数 ----------------------
true_counts = (
    merged["labels_true"]
    .explode()  # 展开所有标签
    .value_counts()  # 计算原始频数
    .reindex(all_labels, fill_value=0)  # 包含所有可能标签
    .astype(int)  # 转换为整数
)
# ----------------------------------------------------------------

# 初始化统计字典
tp = {label: 0 for label in all_labels}
fp = {label: 0 for label in all_labels}
fn = {label: 0 for label in all_labels}

# 逐样本统计
for _, row in merged.iterrows():
    true = row["labels_true"]
    pred = row["labels_pred"]

    for label in all_labels:
        true_has = label in true
        pred_has = label in pred

        if true_has and pred_has:
            tp[label] += 1
        elif pred_has:
            fp[label] += 1
        elif true_has:
            fn[label] += 1

# 计算逐类指标（新增Support列）
metrics = []
for label in all_labels:
    p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0
    r = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0

    metrics.append(
        {
            "Label": label,
            "Precision": round(p, 4),
            "Recall": round(r, 4),
            "F1": round(f1, 4),
            "Occurence": true_counts[label],  # 添加频数字段
        }
    )

# 生成指标表格
metrics_df = pd.DataFrame(metrics)


# 计算宏平均
macro_p = metrics_df["Precision"].mean()
macro_r = metrics_df["Recall"].mean()
macro_f1 = metrics_df["F1"].mean()

# 计算微平均
total_tp = sum(tp.values())
total_fp = sum(fp.values())
total_fn = sum(fn.values())

micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
micro_f1 = (
    2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0
)

# 打印结果
print("逐类指标：")
print(metrics_df)

print("\n宏平均：")
print(f"Precision: {macro_p:.4f}  Recall: {macro_r:.4f}  F1: {macro_f1:.4f}")

print("\n微平均：")
print(f"Precision: {micro_p:.4f}  Recall: {micro_r:.4f}  F1: {micro_f1:.4f}")
