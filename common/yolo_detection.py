# yolo_detection.py

import random

import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


def get_random_color():
    return tuple(random.choices(range(64, 256), k=3))


def draw_boxes(img_pil, bboxes, texts=None, font_size=20):
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("./font/Roboto-Regular.ttf", font_size)
    except:
        font = ImageFont.load_default()
    colors = [get_random_color() for _ in bboxes]
    for i, box in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, box)
        color = colors[i]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        if texts:
            txt = str(texts[i])
            tw, th = draw.textsize(txt, font=font)
            draw.rectangle([x1, y1 - th, x1 + tw + 6, y1], fill=color)
            draw.text((x1 + 3, y1 - th), txt, fill=(0, 0, 0), font=font)
    return img_pil


class YoloDetector:
    def __init__(
        self,
        model_path="/kaggle/input/yolo11_dectection/pytorch/default/1/best.pt",
        device=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)

    def detect(self, img_path, conf_thres=0.5):
        results = self.model(img_path)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        filtered = [(box, conf) for box, conf in zip(boxes, confs) if conf > conf_thres]
        bboxes = [box for box, _ in filtered]
        return bboxes
