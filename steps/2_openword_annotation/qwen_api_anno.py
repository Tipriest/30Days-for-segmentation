from runpy import run_path
import os
import json
import base64
import utils
from openai import OpenAI
from config.args import parse_args
from tqdm import tqdm
from dotenv import load_dotenv


# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs,
# such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained(
#     pretrained_model_name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct",
#     min_pixels=min_pixels,
#     max_pixels=max_pixels)
def encode_image(_image_path):
    with open(_image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


load_dotenv()  # 自动加载.env文件
my_api_key = ""
try:
    my_api_key = os.environ["API_KEY"]
except KeyError:
    # 提醒更新密钥而非直接崩溃
    utils.send_alert("API_KEY missing! Update .env file!")

if __name__ == "__main__":
    args = parse_args()

    results = []

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
                client = OpenAI(
                    base_url="https://api-inference.modelscope.cn/v1/",
                    api_key=my_api_key,  # Your api key
                )

                # Function to encode the image

                base64_image = encode_image(file_path)

                response = client.chat.completions.create(
                    model="Qwen/Qwen2.5-VL-72B-Instruct",  # ModelScope Model-Id
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": args.question,
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpg;base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                    # stream=True,
                )
                a = response.choices[0].message.content
                result = json.loads(response.choices[0].message.content)
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
