import os
import fiftyone as fo

if __name__ == "__main__":
    # 配置路径
    image_dir = "/home/tipriest/Documents/30Days-for-segmentation/steps\
/1_preprocess/key_frames"  
    groundtruth_csv = "/home/tipriest/Documents/30Days-for-segmentation\
/steps/2_openword_annotation/anno_results/labels_groundtruth.csv"
    pred_result_csv = "/home/tipriest/Documents/30Days-for-segmentation\
/steps/2_openword_annotation/anno_results/labels.csv"

    # 创建数据集
    dataset = fo.Dataset(name="terrain_multi_classification", overwrite=True)
    samples_dict = {}

    # 处理groundtruth.csv
    with open(groundtruth_csv, "r", encoding="utf-8") as f:
        header = f.readline()  # 跳过标题行
        for line in f:
            line = line.strip() #能够去掉换行
            if not line:
                continue

            # 分割第一个空格作为文件名和标签
            frame, labels_str = line.split(",", 1)
            labels = [l.strip() for l in labels_str.split(",")]
            labels = sorted(labels)

            # 创建样本
            sample = fo.Sample(filepath=os.path.join(image_dir, frame))
            sample["ground_truth"] = fo.Classifications(
                classifications=[fo.Classification(label=l) for l in labels]
            )
            samples_dict[frame] = sample

    # 添加样本到数据集
    dataset.add_samples(list(samples_dict.values()))

    # 处理pred_result.csv
    with open(pred_result_csv, "r", encoding="utf-8") as f:
        header = f.readline()  # 跳过标题行
        for line in f:
            line = line.strip()
            if not line:
                continue

            frame, labels_str = line.split(",", 1)
            labels = [l.strip() for l in labels_str.split(",")]
            labels = sorted(labels)

            if frame in samples_dict:
                sample = samples_dict[frame]
                sample["predictions"] = fo.Classifications(
                    classifications=[fo.Classification(label=l) for l in labels]
                )
                sample.save()

    # 验证数据集
    print(dataset)
    print("样本数量:", len(dataset))

    # 启动可视化界面
    session = fo.launch_app(dataset)
    session.wait()
