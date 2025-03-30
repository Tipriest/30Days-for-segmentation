from runpy import run_path
import os
import json
import base64
import utils
from openai import OpenAI
from config.args import parse_args
from tqdm import tqdm
from dotenv import load_dotenv


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
    min_pixels=512 * 28 * 28,
    max_pixels=2048 * 28 * 28,
):
    _base64_image = encode_image(image_path)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
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
    if args.input_dir:
        from glob import glob

        image_extensions = ["jpg", "jpeg", "png", "bmp"]
        for ext in image_extensions:
            image_paths.extend(glob(os.path.join(args.input_dir, f"*.{ext}")))
    elif args.pic_paths:
        image_paths = args.pic_paths

    with utils.timer():
        for idx, file_path in tqdm(enumerate(image_paths, 1)):
            try:
                response = inference_with_api(
                    image_path=file_path,
                    _prompt=args.question1,
                    _client=client,
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
                "./steps/2_openword_annotation/anno_results/labels.csv"
            )
            utils.save_to_csv(results, output_filename)
            print(f"Saved {len(results)} results to {output_filename}")
        else:
            print("No result to save")

    run_path("./steps/2_openword_annotation/anno_results/calF1.py")
