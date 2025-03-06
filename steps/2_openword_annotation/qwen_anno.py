import argparse
import json
from runpy import run_path
import torch
import utils

from qwen_vl_utils import process_vision_info
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    # AutoTokenizer,
    AutoProcessor,
)


# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs,
# such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained(
#     pretrained_model_name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct",
#     min_pixels=min_pixels,
#     max_pixels=max_pixels)


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
        default="/home/tipriest/Documents/30Days-for-segmentation/steps/1_preprocess/key_frames",
        help="Directory containing images",
    )
    parser.add_argument(
        "-q",
        "--question",
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
        default="""Here is a list of terrain related annotations:\
["cement road", "red paved path", "yellow paved path", "soil",\
"lawn", "water", "curb", "others",],\
Determine which annotations of this list occured in this picture?\
You could give multi annotations in the annotation list if there exists multi kind of terrains.\
Use format like this {"annotations": ["item1", "item2", ...]} to return the result data.\
mention don't use other annotations beyond this list.\
mention don't split the phrases like "red paved path" for "paved path" in the annotation list.\
mention don't use synonyms to replace the phrases like "grass" for "lawn" in the annotation list.\
""",
        # default="""
        #             As a professional topography analysis assistant,
        #             please analyze images strictly according to the following rules:
        #             1. Terrain category definition:
        #             - Road category:
        #             * cement road: grey continuous hardening road surface,
        #               which may contain obvious seams or cracks
        #             * red paved path: paved road with red bricks arranged regularly
        #             * yellow paved path: paved road with bright yellow bricks and
        #               possibly diamond-shaped arrangements
        #             * curb: The raised strip structure at the edge of the road,
        #               which usually forms a height difference of 5-15cm from the road surface.
        #             - Nature:
        #             * soil: bare soil or sand, no vegetation coverage
        #             * lawn: A grassy region
        #             * water: liquid water surface (including accumulated water, rivers, etc.)
        #             * others: Special terrain that is obviously not part of the above category
        #             2. Analysis steps:
        #               Focus on observing ground areas and ignore non-terrain elements
        #               such as buildings and sky, Identify the material characteristics
        #               (color, texture, structure) in each area
        #              For questions:
        #             - The color characteristics of the paved path must be clearly defined
        #             3. Output requirements:
        #             - Use strict JSON format
        #             - Empty results return to empty list
        #             - No comment description
        #             - return annotations only in the list of
        #               ["cement road", "red paved path", "yellow paved path", "soil", "lawn",
        #               "water", "curb", "others"], some words in this list like red paved path
        #               shuould be considered as an entity and could not be seperated,
        #               don't use other annotations, this is vital important.
        #             Please return the JSON result that meets the above criteria and
        #               follow the following example to return
        #             {"annotations": ["soil", "lawn", "yellow paved path", ...]}
        #             """,
        # default="你是一个地形分类数据集的专业标注员, 判断一下图片中应该给那些标注? 以csv格式输出标注结果。",
        # default="""
        #         You are a professional annotator for terrain classification data sets.
        #         describe the terrain occured in this picture, and summary the terrain in
        #         csv format like this ["red paved path", "curb", ...]
        #         """,
        help="questions",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # 获取图片链接列表
    image_paths = []
    results = []
    if args.input_dir:
        import os
        from glob import glob

        image_extensions = ["jpg", "jpeg", "png", "bmp"]
        for ext in image_extensions:
            image_paths.extend(glob(os.path.join(args.input_dir, f"*.{ext}")))
    elif args.pic_paths:
        image_paths = args.pic_paths

    # default: Load the model on the available device(s)
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "/home/tipriest/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct",
    #     torch_dtype="auto",
    #     device_map="auto",
    # )

    # We recommend enabling flash_attention_2 for better acceleration and memory saving,
    # especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "/home/tipriest/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    # default processer
    processor = AutoProcessor.from_pretrained(
        "/home/tipriest/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct"
    )
    for idx, file_path in enumerate(image_paths, 1):
        try:
            print(f"Processing {idx}/{len(image_paths)}: {file_path}")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{file_path}"},
                        {"type": "text", "text": args.question},
                    ],
                }
            ]
            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            print(output_text)
            result = json.loads(output_text[0])
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
        output_filename = "./steps/2_openword_annotation/labels.csv"
        utils.save_to_csv(results, output_filename)
        print(f"Saved {len(results)} results to {output_filename}")
    else:
        print("No result to save")

    run_path("./steps/2_openword_annotation/calF1.py")
