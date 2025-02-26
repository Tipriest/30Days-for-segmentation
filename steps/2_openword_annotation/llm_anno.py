# 确定当前环境
# import sys
import argparse
import base64
from io import BytesIO
import csv
import json
from datetime import datetime

from IPython.display import HTML, display
from PIL import Image
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser


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
            writer.writerow([_result["image_path"], ",".join(_result["annotations"])])


def convert_to_base64(_pil_image):
    buffered = BytesIO()
    if _pil_image.mode == "RGBA":
        _pil_image = _pil_image.convert("RGB")
    _pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def plt_img_base64(img_base64):
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    display(HTML(image_html))


def prompt_func(data):
    text = data["text"]
    image = data["image"]

    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpg;base64,{image}",
    }

    content_parts = []

    text_part = {"type": "text", "text": text}

    content_parts.append(image_part)
    content_parts.append(text_part)

    return [HumanMessage(content=content_parts)]


def parse_args():
    parser = argparse.ArgumentParser(description="langchain_test")
    parser.add_argument(
        "-p",
        "--pic_paths",
        nargs="+",
        help="List of picture paths",
    )
    parser.add_argument(
        "-d",
        "--input_dir",
        default="./steps/1_preprocess/key_frames",
        help="Directory containing images",
    )
    parser.add_argument(
        "-q",
        "--question",
        # action="store_true",
        # default="""You are a professional annotator for terrain classification data sets.
        #             Here is a list of annotations:
        #             ["cement road", "red brick road", "yellow brick road", "soil",
        #             "lawn", "water", "curb", "others",],
        #             Determine which annotations should be given in the picture?
        #             Output the annotation result in csv format like this ["red brick road", "curb", ...]
        #             Caution don't output other things, just the annotation result, thanks
        #         """,
        default="""You are a professional annotator for terrain classification data sets.
                    Here is a list of annotations:
                    ["cement road", "red brick road", "yellow brick road", "soil",
                    "lawn", "water", "curb", "others",],
                    Determine which annotations should be given in the picture?
                    Use Json format like this {"annotations": ["soil"]} to return the result data.
                """,
        # default="你是一个地形分类数据集的专业标注员, 判断一下图片中应该给那些标注? 以csv格式输出标注结果。",
        # default="""
        #         You are a professional annotator for terrain classification data sets.
        #         describe the terrain occured in this picture, and summary the terrain in
        #         csv format like this ["red brick road", "curb", ...]
        #         """,
        help="questions",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = []

    # 获取图片链接列表
    image_paths = []
    if args.input_dir:
        import os
        from glob import glob

        image_extensions = ["jpg", "jpeg", "png", "bmp"]
        for ext in image_extensions:
            image_paths.extend(glob(os.path.join(args.input_dir, f"*.{ext}")))
    elif args.pic_paths:
        image_paths = args.pic_paths

    # 定义模型
    llm = ChatOllama(
        model="llava",
        # model="deepseek-r1:14b",
        temperature=0.2,
        format="json",
    )

    chain = prompt_func | llm | StrOutputParser()

    # 处理每张图片
    # FIXME: 这里为什么指定enumerate的start是1?
    for idx, file_path in enumerate(image_paths, 1):
        try:
            print(f"Processing {idx}/{len(image_paths)}: {file_path}")
            pil_image = Image.open(file_path)
            image_b64 = convert_to_base64(pil_image)

            response = chain.invoke(
                {
                    "text": args.question,
                    "image": image_b64,
                }
            )
            result = json.loads(response)
            result = result["annotations"]
            results.append(
                {
                    "image_path": file_path.split("/")[-1],
                    "annotations": result[:],
                }
            )

        except Exception as e:
            print(f"Error processing {file_path} : {str(e)}")
            results.append(
                {
                    "image_path": file_path,
                    "annotations": f"ERROR: {str(e)}",
                }
            )

    if results:
        output_filename = (
            f"annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        save_to_csv(results, output_filename)
        print(f"Saved {len(results)} results to {output_filename}")
    else:
        print("No result to save")
