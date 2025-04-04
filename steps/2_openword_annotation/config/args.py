import argparse


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
        "-d2",
        "--input_dir2",
        default="/data/Competitions/images/train",
        help="Directory containing images",
    )

    parser.add_argument(
        "-q1",
        "--question1",
        # prompt1
        default="""You are a professional annotator for terrain classification data sets.\
Here is a list of annotations:\
["cement road", "red paved path", "yellow paved path", "soil", \
"lawn", "water", "curb", "others",],\
Determine the annotations should be given for this picture.
You could give multi annotations in the annotation list if there exists multi kind of terrains.\
Use format like this {"annotations": ["item1", "item2", ...]} to return the result data.\
mention don't use other annotations beyond this list.\
mention don't split the phrases like "red paved path" for "paved path" in the annotation list.\
mention don't use synonyms to replace the phrases like "grass" for "lawn" in the annotation list.\
""",
        help="questions1",
    )

    parser.add_argument(
        "-q2",
        "--question2",
        # prompt2
        default="""Here is a list of terrain related annotations:\
["cement road", "red paved path", "yellow paved path", "soil",\
"lawn", "water", "curb", "others",],\
Determine which annotations of this list occured in this picture?\
You could give multi annotations in the annotation list if there exists multi kind of terrains.\
Use format like this {"annotations": ["item1", "item2", ...]} to return the result data. \
mention don't use other annotations beyond this list. \
mention don't split the phrases like "red paved path" for "paved path" in the annotation list. \
mention don't use synonyms to replace the phrases like "grass" for "lawn" in the annotation list. \
""",
        help="questions",
    )
    parser.add_argument(
        "-q3",
        "--question3",
        default="""
                    As a professional topography analysis assistant,
                    please analyze images strictly according to the following rules:
                    1. Terrain category definition:
                    - Road category:
                    * cement road: grey continuous hardening road surface,
                      which may contain obvious seams or cracks
                    * red paved path: paved road with red bricks arranged regularly
                    * yellow paved path: paved road with bright yellow bricks and
                      possibly diamond-shaped arrangements
                    * curb: The raised strip structure at the edge of the road,
                      which usually forms a height difference of 5-15cm from the road surface.
                    - Nature:
                    * soil: bare soil or sand, no vegetation coverage
                    * lawn: A grassy region
                    * water: liquid water surface (including accumulated water, rivers, etc.)
                    * others: Special terrain that is obviously not part of the above category
                    2. Analysis steps:
                      Focus on observing ground areas and ignore non-terrain elements
                      such as buildings and sky, Identify the material characteristics
                      (color, texture, structure) in each area
                     For questions:
                    - The color characteristics of the paved path must be clearly defined
                    3. Output requirements:
                    - Use strict JSON format
                    - Empty results return to empty list
                    - No comment description
                    - return annotations only in the list of
                      ["cement road", "red paved path", "yellow paved path", "soil", "lawn",
                      "water", "curb", "others"], some words in this list like red paved path
                      shuould be considered as an entity and could not be seperated,
                      don't use other annotations, this is vital important.
                    Please return the JSON result that meets the above criteria and
                      follow the following example to return
                    {"annotations": ["soil", "lawn", "yellow paved path", ...]}
                    """,
        # default="你是一个地形分类数据集的专业标注员, 判断一下图片中应该给那些标注? 以csv格式输出标注结果。",
        # default="""
        #         You are a professional annotator for terrain classification data sets.
        #         describe the terrain occured in this picture, and summary the terrain in
        #         csv format like this ["red paved path", "curb", ...]
        #         """,
        help="questions3",
    )

    parser.add_argument(
        "-q4",
        "--question4",
        default="""从消防安全的角度框出走廊中危险物品的位置，比如易燃物品和阻塞了消防安全通道的物品, 以json格式输出所有的坐标""",
        help="questions4",
    )

    parser.add_argument(
        "-vlq",
        "--vlm_question",

        ## prompt2
        default="""Here is a list of terrain related annotations:\
["cement road", "red paved path", "yellow paved path", "soil",\
"lawn", "water", "curb", "others",],\
Try to describe what kind of terrains existed in this picture.
""",
        help="questions",
    )

    parser.add_argument(
        "--llm_question",


        default="""Here is a list of terrain related annotations:\
["cement road", "red paved path", "yellow paved path", "soil",\
"lawn", "water", "curb", "others",],\
Determine which annotations of this list occured through the description?\
You could give multi annotations in the annotation list if there exists multi kind of terrains.\
Use format like this {"annotations": ["item1", "item2", ...]} to return the result data. \
mention don't use other annotations beyond this list. \
mention don't split the phrases like "red paved path" for "paved path" in the annotation list. \
mention don't use synonyms to replace the phrases like "grass" for "lawn" in the annotation list. \
""",
        help="llm_question",
    )

    return parser.parse_args()
