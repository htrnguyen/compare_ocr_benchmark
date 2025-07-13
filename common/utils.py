import os

import pandas as pd


def read_annotations(csv_path):
    return pd.read_csv(csv_path)


def save_results(results, output_csv):
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Lưu thành công: {output_csv}")


def get_all_image_paths(img_dir, extensions=[".jpg", ".png", ".jpeg"]):
    paths = []
    for fname in os.listdir(img_dir):
        if any(fname.lower().endswith(ext) for ext in extensions):
            paths.append(os.path.join(img_dir, fname))
    return paths
