# 确定当前环境
# import sys
import json
from runpy import run_path

# from datetime import datetime
import utils
from IPython.display import HTML, display
from PIL import Image
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from config.args import parse_args


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
            image_b64 = utils.convert_to_base64(pil_image)

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
        output_filename = (
            "./steps/2_openword_annotation/anno_results/labels.csv"
        )
        utils.save_to_csv(results, output_filename)
        print(f"Saved {len(results)} results to {output_filename}")
    else:
        print("No result to save")

    run_path("./steps/2_openword_annotation/anno_results/calF1.py")
