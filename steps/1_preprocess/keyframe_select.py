import os
import cv2
import numpy as np
import shutil

prev_frame = None
# key_frames = []
print(os.getcwd())
for frame_path in sorted(os.listdir("./steps/1_preprocess/frames")):
    frame = cv2.imread(
        os.path.join(os.getcwd(), "./steps/1_preprocess/frames", frame_path)
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_frame is not None:
        diff = cv2.absdiff(gray, prev_frame)
        if np.mean(diff) > 30:  # 差异阈值，根据场景调整
            # key_frames.append(frame_path)
            shutil.copyfile(
                os.path.join(
                    os.getcwd(), "./steps/1_preprocess/frames", frame_path
                ),
                os.path.join(
                    os.getcwd(), "./steps/1_preprocess/key_frames", frame_path
                ),
            )
    else:
        shutil.copyfile(
            os.path.join(
                os.getcwd(), "./steps/1_preprocess/frames", frame_path
            ),
            os.path.join(
                os.getcwd(), "./steps/1_preprocess/key_frames", frame_path
            ),
        )
    prev_frame = gray
