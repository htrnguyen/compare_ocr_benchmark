import re

import Levenshtein
from jiwer import cer, wer


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(
        r"[^a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s.,!?]",
        "",
        text,
    )
    return text.strip()


def compute_metrics(gt, pred):
    gt_norm = normalize_text(gt)
    pred_norm = normalize_text(pred)
    return {
        "cer": cer(gt_norm, pred_norm),
        "wer": wer(gt_norm, pred_norm),
        "lev": Levenshtein.distance(gt_norm, pred_norm),
        "acc": int(gt_norm == pred_norm),
    }
