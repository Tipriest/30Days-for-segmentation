# 确定当前环境
# import sys
import argparse
import base64
from io import BytesIO
import csv
import json
from runpy import run_path

# from datetime import datetime

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
            writer.writerow(
                [_result["image_path"], ",".join(_result["annotations"])]
            )


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


def prompt_func_vlm(data):
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


def prompt_func_llm(data):
    text = data["text"]

    content_parts = []

    text_part = {"type": "text", "text": text}

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
        "-vq",
        "--vlm_question",
        # action="store_true",
        # default="""You are a professional annotator for terrain classification data sets.
        #             Here is a list of annotations:
        #             ["cement road", "red paved path", "yellow paved path", "soil",
        #             "lawn", "water", "curb", "others",],
        #             Determine which annotations should be given in the picture?
        #             Output the annotation result in csv format like this:
        #             ["red paved path", "curb", ...]
        #             Caution don't output other things, just the annotation result, thanks
        #         """,
        default="""Use no more than 200 words to describe the contents of this image \
in detail from the perspective of the terrain.
especially if there exists ["cement road", "red paved path", "yellow paved path", "soil",\
"lawn", "water", "curb", "others",], describe the colors please. 
""",
        help="vlm questions",
    )
    parser.add_argument(
        "-lq",
        "--llm_question",
        # action="store_true",
        default="""Here is a list of terrain related annotations:\
["cement road", "red paved path", "yellow paved path", "soil",\
"lawn", "water", "curb", "others",],\
Determine which annotations of this list occured in this picture?\
You could give multi annotations in the annotation list if there exists multi kind of terrains.\
Use Json format like this {"annotations": []} to return the result data.\
mention you could give multiple annotations if there exists multiple kinds of terrains.\
mention don't split the phrases like "red paved path" to "paved path" in the annotation list.\
mention don't use synonyms to replace the phrases like "grass" for "lawn" in the annotation list.\
""",
        help="llm questions",
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
    vlm = ChatOllama(
        model="llava:13b",
        # model="deepseek-r1:14b",
        temperature=0.5,
        # format="json",
    )

    vlm_json = ChatOllama(
        model="llava:13b",
        # model="deepseek-r1:14b",
        temperature=0.5,
        format="json",
    )

    llm = ChatOllama(
        # model="llava:13b",
        model="deepseek-r1:14b",
        temperature=0.5,
        format="json",
    )

    chain_vlm = prompt_func_vlm | vlm | StrOutputParser()
    chain_llm = prompt_func_llm | vlm_json | StrOutputParser()

    # 处理每张图片
    # 这里为什么指定enumerate的start是1?
    for idx, file_path in enumerate(image_paths, 1):
        try:
            print(f"Processing {idx}/{len(image_paths)}: {file_path}")
            pil_image = Image.open(file_path)
            image_b64 = convert_to_base64(pil_image)

            vlm_response = chain_vlm.invoke(
                {
                    "text": args.vlm_question,
                    "image": image_b64,
                }
            )
            llm_response = chain_llm.invoke({"text": args.llm_question})

            result = json.loads(llm_response)
            result = result["annotations"]
            results.append(
                {
                    "image_path": file_path.split("/")[-1],
                    "annotations": result[:],
                }
            )
            print(vlm_response)
            print(llm_response)

        except Exception as e:
            print(f"Error processing {file_path} : {str(e)}")
            results.append(
                {
                    "image_path": file_path,
                    "annotations": f"ERROR: {str(e)}",
                }
            )

    if results:
        output_filename = "./steps/2_openword_annotation/labels.csv"
        save_to_csv(results, output_filename)
        print(f"Saved {len(results)} results to {output_filename}")
    else:
        print("No result to save")

    run_path("./steps/2_openword_annotation/calF1.py")
