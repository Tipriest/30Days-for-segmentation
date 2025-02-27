# 30 Days Plan for semantic segmentation

## 0. Questions

- 弱监督语义分割到底怎么用于场景分割，比如自然场景?
- 现在的这个代码在猫啊狗啊的数据集上试一下
- 生成其他种类的弱监督标签试一下
- 一些新的想法，拿VLM先尽可能详细地描述一下，把response传到LLM中，让LLM选择词
- 拿很多张VLM的结果传到LLM中来判断一个视频中出现了哪些地形
- 拿视频理解的model理解一个视频中出现了哪些地形

## 1. How to start

```bash
git clone https://github.com/Tipriest/30Days-for-segmentation.git
git submodule update --init --recursive

# Set Conda Env
conda env create -f environment.yml
conda activate 30daysseg
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e .

# Use FCN
cd ./classical_solution/fcn
pip install .
```

## 2. My steps

### 2.1 添加视频material软链接

```bash
cd ./steps/0_material
sudo ln -s /home/tipriest/data/TerrainSeg/hit_grass/videos_record/VID_20220502_135318.mp4 ./original_video.mp4
```

### 2.2 采用FFmpeg提取帧

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# 提取视频帧
mkdir ../1_preprocess/frames
ffmpeg -i original_video.mp4 -vf "fps=5" ../1_preprocess/frames/%06d.jpg

# 提取视频关键帧
python keyframes_select.py


```

### 2.3 采用CLIP进行标注

```bash
pip install git+https://github.com/openai/CLIP.git


```
