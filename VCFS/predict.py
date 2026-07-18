"""Interactive inference utilities for VCFS checkpoints."""

import os
import time

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from deeplab import DeeplabV3
from utils.class_config import load_class_config


if __name__ == "__main__":
    class_config = load_class_config()
    deeplab = DeeplabV3()

    mode = "predict"

    count = False
    name_classes = class_config["classes"]

    video_path = 0
    video_save_path = ""
    video_fps = 25.0

    test_interval = 100
    fps_image_path = "img/street.jpg"

    dir_origin_path = "FacadeWHU_origin/JPEGImages"
    dir_save_path = "paper_out_facadeWHU/"

    simplify = True
    onnx_save_path = "model_data/models.onnx"

    if mode == "predict":
        while True:
            img = input("Input image filename:")
            try:
                image = Image.open(img)
            except Exception:
                print("Open Error! Try again!")
                continue
            r_image = deeplab.detect_image(image, count=count, name_classes=name_classes)
            r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            size = (
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("Could not read from the camera or video source.")

        fps = 0.0
        while True:
            t1 = time.time()
            ref, frame = capture.read()
            if not ref:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(deeplab.detect_image(frame))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1.0 / (time.time() - t1))) / 2
            print("fps= %.2f" % fps)
            frame = cv2.putText(
                frame,
                "fps= %.2f" % fps,
                (0, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("video", frame)
            key = cv2.waitKey(1) & 0xFF
            if video_save_path != "":
                out.write(frame)

            if key == 27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = deeplab.get_FPS(img, test_interval)
        print(str(tact_time) + " seconds, " + str(1 / tact_time) + "FPS, @batch_size 1")

    elif mode == "dir_predict":
        img_names = os.listdir(dir_origin_path)
        os.makedirs(dir_save_path, exist_ok=True)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                (".bmp", ".dib", ".png", ".jpg", ".jpeg", ".pbm", ".pgm", ".ppm", ".tif", ".tiff")
            ):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = deeplab.detect_image(image)
                r_image.save(os.path.join(dir_save_path, img_name))
    elif mode == "export_onnx":
        deeplab.convert_to_onnx(simplify, onnx_save_path)

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
