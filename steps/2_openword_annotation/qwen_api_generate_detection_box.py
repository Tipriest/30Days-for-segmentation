import os
import base64
import ast
import shutil

from openai import OpenAI
import utils
from config.args import parse_args
from tqdm import tqdm
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont, ImageColor
from qwen_vl_utils import smart_resize


additional_colors = [
    colorname for (colorname, colorcode) in ImageColor.colormap.items()
]


def encode_image(_image_path):
    with open(_image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# 加载API Key
def load_apikey():
    load_dotenv()  # 自动加载.env文件
    _my_api_key = ""
    try:
        _my_api_key = os.environ["API_KEY"]
    except KeyError:
        # 提醒更新密钥而非直接崩溃
        utils.send_alert("API_KEY missing! Update .env file!")
    return _my_api_key


# @title inference function with API
def inference_with_api(
    image_path,
    _prompt,
    _client,
    sys_prompt="You are a helpful assistant.",
    model_id="Qwen/Qwen2.5-VL-72B-Instruct",
    _min_pixels=512 * 28 * 28,
    _max_pixels=2048 * 28 * 28,
):
    _base64_image = encode_image(image_path)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "min_pixels": _min_pixels,
                    "max_pixels": _max_pixels,
                    # Pass in BASE64 image data. Note that the image format
                    # (i.e., image/{format}) must match the Content Type
                    # in the list of supported images. "f" is the method
                    # for string formatting.
                    # PNG image:  f"data:image/png; base64,{base64_image}"
                    # JPEG image: f"data:image/jpeg;base64,{base64_image}"
                    # WEBP image: f"data:image/webp;base64,{base64_image}"
                    # JPG image: f"data:image/jpg;  base64,{base64_image}"
                    "image_url": {
                        "url": f"data:image/jpg;base64,{_base64_image}"
                    },
                },
                {"type": "text", "text": _prompt},
            ],
        },
    ]
    completion = _client.chat.completions.create(
        model=model_id,
        messages=messages,
    )
    return completion.choices[0].message.content


# @title Parsing JSON output
def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(
                lines[i + 1 :]
            )  # Remove everything before "```json"
            json_output = json_output.split("```", maxsplit=1)[
                0
            ]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output

def image_to_label_path(_image_path):
    """
    将图片路径转换为YOLO标签路径
    示例输入：/data/Competitions/images/train/wtz (9).jpg
    示例输出：/data/Competitions/labels/train/wtz (9).txt
    """
    # 拆分路径和文件名（参考）
    _dir_path, _filename = os.path.split(_image_path)

    # 替换目录层级（参考）
    new_dir = _dir_path.replace(
        "images", "labels", 1
    )  # 只替换第一个出现的images

    # 修改文件扩展名（参考）
    base_name = list(os.path.splitext(_filename))  # 去掉.jpg扩展名
    new_filename = f"{base_name[0]}.txt"

    # 组合新路径（参考）
    return os.path.join(new_dir, new_filename)


def plot_bounding_boxes(im, bounding_boxes, _input_width, _input_height):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL,
    normalized coordinates, and different colors.

    Args:
        img_path: The path to the image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
    """

    # Load the image
    img = im
    width, height = img.size
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
        "red",
        "green",
        "blue",
        "yellow",
        "orange",
        "pink",
        "purple",
        "brown",
        "gray",
        "beige",
        "turquoise",
        "cyan",
        "magenta",
        "lime",
        "navy",
        "maroon",
        "teal",
        "olive",
        "coral",
        "lavender",
        "violet",
        "gold",
        "silver",
    ] + additional_colors

    # Parsing out the markdown fencing
    bounding_boxes = parse_json(bounding_boxes)

    font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)

    try:
        json_output = ast.literal_eval(bounding_boxes)
    except Exception as e:
        end_idx = bounding_boxes.rfind('"}') + len('"}')
        truncated_text = bounding_boxes[:end_idx] + "]"
        json_output = ast.literal_eval(truncated_text)

    # Iterate over the bounding boxes
    for i, bounding_box in enumerate(json_output):
        # Select a color from the list
        color = colors[i % len(colors)]

        # Convert normalized coordinates to absolute coordinates
        abs_y1 = int(bounding_box["bbox_2d"][1] / _input_height * height)
        abs_x1 = int(bounding_box["bbox_2d"][0] / _input_width * width)
        abs_y2 = int(bounding_box["bbox_2d"][3] / _input_height * height)
        abs_x2 = int(bounding_box["bbox_2d"][2] / _input_width * width)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        # Draw the bounding box
        draw.rectangle(
            ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
        )

        # Draw the text
        if "label" in bounding_box:
            draw.text(
                (abs_x1 + 8, abs_y1 + 6),
                bounding_box["label"],
                fill=color,
                font=font,
            )

    # Display the image
    # img.show()
    return img


def generate_yolo_label(
    _file_path, _bounding_boxes, _input_width, _input_height
):
    # 获取图片尺寸
    img = Image.open(_file_path)
    img_width, img_height = img.size
    # 转换每个检测框
    yolo_lines = []
    # Parsing out the markdown fencing
    _bounding_boxes = parse_json(_bounding_boxes)
    try:
        json_output = ast.literal_eval(_bounding_boxes)
    except Exception:
        end_idx = _bounding_boxes.rfind('"}') + len('"}')
        truncated_text = _bounding_boxes[:end_idx] + "]"
        json_output = ast.literal_eval(truncated_text)

    for _, bounding_box in enumerate(json_output):
        # Convert normalized coordinates to absolute coordinates
        abs_y1 = int(bounding_box["bbox_2d"][1] / _input_height * img_height)
        abs_x1 = int(bounding_box["bbox_2d"][0] / _input_width * img_width)
        abs_y2 = int(bounding_box["bbox_2d"][3] / _input_height * img_height)
        abs_x2 = int(bounding_box["bbox_2d"][2] / _input_width * img_width)
        cls_id = 0
        x_center = (abs_x1 + abs_x2) / 2 / img_width
        y_center = (abs_y1 + abs_y2) / 2 / img_height
        _width = (abs_x2 - abs_x1) / img_width
        _height = (abs_y2 - abs_y1) / img_height
        yolo_lines.append(
            f"{cls_id} {x_center:.6f} {y_center:.6f} {_width:.6f} {_height:.6f}"
        )
        # 保存到对应txt文件
        with open(image_to_label_path(_file_path), "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))


if __name__ == "__main__":
    args = parse_args()

    results = []
    client = OpenAI(
        # If the environment variable is not configured,
        # please replace the following line with the Dashscope API Key:
        # api_key="sk-xxx".
        api_key=load_apikey(),
        base_url="https://api-inference.modelscope.cn/v1/",
    )
    # 获取图片链接列表
    image_paths = []
    input_dir = args.input_dir2

    if input_dir:
        from glob import glob

        image_extensions = ["jpg", "jpeg", "png", "bmp"]
        for ext in image_extensions:
            image_paths.extend(glob(os.path.join(input_dir, f"*.{ext}")))
    elif args.pic_paths:
        image_paths = args.pic_paths

    output_dir = input_dir.replace("images", "labels", 1)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # 递归删除非空文件夹
        print(f"delete old dir: {output_dir}")
    # 创建新文件夹（支持多级目录创建，参考）
    os.makedirs(output_dir, exist_ok=True)
    print(f"make new dir: {output_dir}")

    with utils.timer():
        for idx, img_path in tqdm(enumerate(image_paths, 1)):
            _, filename = os.path.split(img_path)
            if filename.startswith('wtz'):
                continue
            try:
                image = Image.open(img_path)
                width, height = image.size
                min_pixels = 512 * 28 * 28
                max_pixels = 2048 * 28 * 28
                width, height = image.size
                input_height, input_width = smart_resize(
                    height, width, min_pixels=min_pixels, max_pixels=max_pixels
                )

                response = inference_with_api(
                    image_path=img_path,
                    _prompt=args.question4,
                    _client=client,
                    _min_pixels=min_pixels,
                    _max_pixels=max_pixels,
                )

                output_image = plot_bounding_boxes(
                    image, response, input_width, input_height
                )
                generate_yolo_label(
                    img_path, response, input_width, input_height
                )

            except Exception as e:
                print(f"Error processing {img_path} : {str(e)}")
                # results.append(
                #     {
                #         "image_path": file_path,
                #         "annotations": f"ERROR: {str(e)}",
                #     }
                # )
