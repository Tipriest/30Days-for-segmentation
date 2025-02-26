# 确定当前环境
# import sys
import argparse
import base64
from io import BytesIO

from IPython.display import HTML, display
from PIL import Image
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser


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
        "--pic_path",
        # action="store_true",
        default="/home/tipriest/Documents/30Days-for-segmentation/steps/1_preprocess/key_frames/000111.jpg",
        help="picture_path",
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
        #         """,
        # default="你是一个地形分类数据集的专业标注员, 判断一下图片中应该给那些标注? 以csv格式输出标注结果。",
        default="""
                You are a professional annotator for terrain classification data sets.
                describe the terrain occured in this picture, and summary the terrain in  
                csv format like this ["red brick road", "curb", ...]
                """,
        help="questions",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    file_path = args.pic_path
    pil_image = Image.open(file_path)
    llm = ChatOllama(
        model="llava",
        # model="deepseek-r1:14b",
        temperature=0.5,
    )
    image_b64 = convert_to_base64(pil_image)
    plt_img_base64(image_b64)
    chain = prompt_func | llm | StrOutputParser()
    query_chain = chain.invoke(
        {
            "text": args.question,
            "image": image_b64,
        }
    )
    print(query_chain)
