#!/usr/bin/env python
# coding: utf-8

# In[1]:


import kagglehub

# Download latest version
path = kagglehub.dataset_download("trongnguyen04/nckh-2425-crops")
print("Path to dataset files:", path)

# ─── copy raw folder “1” → /content/dataset_original/crops ───────
SRC="/root/.cache/kagglehub/datasets/trongnguyen04/nckh-2425-crops/versions/1"
DEST="/content/dataset_original/crops"
get_ipython().system('mkdir -p $(dirname $DEST)')
get_ipython().system('cp -r "$SRC" "$DEST"')


# In[2]:


get_ipython().system('git clone https://github.com/htrnguyen/compare_ocr_benchmark.git')


# In[3]:


get_ipython().system('pip install python-Levenshtein jiwer')


# In[5]:


import sys
sys.path.append('/content/compare_ocr_benchmark/common')

from image_preprocessing import preprocess_image
from ocr_base import OCRModelBase
from metrics import compute_metrics
from utils import read_annotations, save_results

import os
import pandas as pd
import time


# In[6]:


get_ipython().system('pip install transformers==4.37.2 timm einops')


# In[7]:


from transformers import AutoModel, AutoTokenizer
import torch
import torchvision.transforms as T
from PIL import Image


# In[8]:


class VinternOCRModel(OCRModelBase):
    """
    Class VinternOCRModel cho phép tùy biến prompt theo từng label.
    Tự động chọn prompt theo từng nhãn cho inference tối ưu.
    """
    def __init__(self, model_name="5CD-AI/Vintern-1B-v3_5", device="cuda"):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        import torch
        import torchvision.transforms as T

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval().to(device)
        self.device = device
        self.transform = self.build_transform(448)
        self.prompts = {
            "brand":
                "Trích xuất chính xác TÊN THƯƠNG HIỆU (brand) duy nhất xuất hiện trên ảnh. "
                "Chỉ trả về một chuỗi ký tự duy nhất (brand). Đáp án định dạng JSON: {\"text\": \"...\"}. "
                "Nếu không nhận diện được, trả về: {\"text\": \"None\"}. Tuyệt đối không giải thích gì thêm.",
            "name":
                "Trích xuất chính xác TÊN SẢN PHẨM (name) duy nhất xuất hiện trên ảnh. "
                "Chỉ trả về một chuỗi ký tự duy nhất (name). Đáp án định dạng JSON: {\"text\": \"...\"}. "
                "Nếu không nhận diện được, trả về: {\"text\": \"None\"}. Tuyệt đối không giải thích gì thêm.",
            "date":
                "Trích xuất chính xác NGÀY THÁNG (date) xuất hiện trên ảnh (ưu tiên dạng dd/mm/yyyy hoặc yyyy-mm-dd). "
                "Chỉ trả về một chuỗi ký tự duy nhất (date). Đáp án định dạng JSON: {\"text\": \"...\"}. "
                "Nếu không nhận diện được, trả về: {\"text\": \"None\"}. Tuyệt đối không giải thích gì thêm.",
            "weight":
                "Trích xuất chính xác KHỐI LƯỢNG (weight, bao gồm cả số và đơn vị: g, kg, ml...) xuất hiện trên ảnh. "
                "Chỉ trả về một chuỗi ký tự duy nhất (weight). Đáp án định dạng JSON: {\"text\": \"...\"}. "
                "Nếu không nhận diện được, trả về: {\"text\": \"None\"}. Tuyệt đối không giải thích gì thêm.",
            "default":
                "Trích xuất toàn bộ văn bản (bao gồm cả chữ và số) xuất hiện trong ảnh này. "
                "Chỉ trả về kết quả ở định dạng JSON với khóa 'text'. "
                "Nếu không nhận diện được văn bản nào, trả về: {\"text\": \"None\"}. "
                "Tuyệt đối không giải thích, không bổ sung bất cứ thông tin nào ngoài JSON."
        }

    def build_transform(self, input_size=448):
        import torchvision.transforms as T
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def extract_text_from_json(self, response_text):
        import json
        try:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            json_str = response_text[start:end]
            out = json.loads(json_str)
            return out.get("text", "None")
        except Exception:
            return "None"

    def get_prompt(self, label):
        """
        Lấy prompt phù hợp với label.
        Nếu label không thuộc tập đã định nghĩa thì dùng prompt default.
        """
        if label is None or label not in self.prompts:
            return self.prompts["default"]
        return self.prompts[label]

    def predict(self, img_pil, img_np=None, label=None):
        """
        img_pil: ảnh PIL đã preprocess.
        label: nhãn để chọn prompt phù hợp.
        """
        import torch, time
        pixel_values = self.transform(img_pil).unsqueeze(0).to(torch.bfloat16).to(self.device)
        prompt = self.get_prompt(label)
        t1 = time.perf_counter()
        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                dict(
                    max_new_tokens=256,
                    do_sample=False,
                    num_beams=3,
                    repetition_penalty=3.5
                )
            )
        t2 = time.perf_counter()
        pred = self.extract_text_from_json(response.strip())
        return pred, t2 - t1


# In[9]:


CSV_ANN = '/content/dataset_original/crops/crops_gt.csv'
IMG_DIR = '/content/dataset_original/crops'

df = read_annotations(CSV_ANN)


# In[10]:


model = VinternOCRModel()

results = []
for idx, row in df.iterrows():
    fname = row['filename']
    desc_gt = row['description_gt']
    label = row.get('label', '')
    img_path = os.path.join(IMG_DIR, fname)

    img_pil, img_np = preprocess_image(img_path, target_size=448, do_auto=True)
    pred, infer_time = model.predict(img_pil, img_np)
    metrics = compute_metrics(desc_gt, pred)

    results.append({
        "filename": fname,
        "label": label,
        "ground_truth": desc_gt,
        "predicted_text": pred,
        "cer": metrics["cer"],
        "wer": metrics["wer"],
        "lev": metrics["lev"],
        "acc": metrics["acc"],
        "time": round(infer_time, 3)
    })


# In[11]:


import os
OUT_CSV = '/content/compare_ocr_benchmark/results/vintern_results.csv'
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
save_results(results, OUT_CSV)

