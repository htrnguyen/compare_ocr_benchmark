#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('git clone https://github.com/htrnguyen/vietocr.git')


# In[2]:


import sys, os, time
import torch
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append('/kaggle/working/vietocr')
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

# Đảm bảo dùng GPU nếu có
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Đường dẫn dữ liệu và model
DATA_ROOT = '/kaggle/input/nckh-2425-crops'
CSV_ANN = '/kaggle/input/nckh-2425-crops/crops_gt.csv'
WEIGHT_PATH = '/kaggle/input/vietocr_fineturning/gguf/default/1/transformerocr.pth'


# In[3]:


def extract_vocab_from_labels(label_files):
    chars = set()
    for fpath in label_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    _, label = line.strip().split("\t")
                    chars.update(label)
                except:
                    continue
    return "".join(sorted(chars))

vocab = extract_vocab_from_labels([
    '/kaggle/input/vietocr-data-pretrain/dataset/test_annotation.txt',
    '/kaggle/input/vietocr-data-pretrain/dataset/train_annotation.txt'
])


# In[4]:


config = Cfg.load_config_from_name("vgg_transformer")
config["vocab"] = vocab
config["weights"] = WEIGHT_PATH
config["device"] = device
predictor = Predictor(config)
config


# In[5]:


get_ipython().system('git clone https://github.com/htrnguyen/compare_ocr_benchmark.git')


# In[6]:


get_ipython().system('pip install python-Levenshtein jiwer')


# In[7]:


import sys
sys.path.append('/kaggle/working/compare_ocr_benchmark/common')
from metrics import compute_metrics
from utils import save_results


# In[8]:


df = pd.read_csv(CSV_ANN)

results = []
for idx, row in df.iterrows():
    fname = row['filename']
    desc_gt = row['description_gt']
    label = row.get('label', '')
    img_path = os.path.join(DATA_ROOT, fname)

    try:
        image = Image.open(img_path).convert("RGB")
        t1 = time.perf_counter()
        pred = predictor.predict(image)
        t2 = time.perf_counter()
        infer_time = round(t2 - t1, 3)
    except Exception as e:
        pred = f"OCR_Error: {e}"
        infer_time = 0.0

    # Dùng hàm metrics chung (import từ compare_ocr_benchmark/common/metrics.py)
    from metrics import compute_metrics
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
        "time": infer_time
    })
    if idx % 50 == 0:
        print(f"Processed {idx}/{len(df)}")


# In[9]:


import os
OUT_CSV = '/kaggle/working/compare_ocr_benchmark/results/vietocr_pretrain_results.csv'
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
save_results(results, OUT_CSV)


# In[ ]:




