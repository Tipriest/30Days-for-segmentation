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
        output_filename = (
            "./steps/2_openword_annotation/anno_results/labels.csv"
        )
        utils.save_to_csv(results, output_filename)
        print(f"Saved {len(results)} results to {output_filename}")
    else:
        print("No result to save")

    run_path("./steps/2_openword_annotation/anno_results/calF1.py")
