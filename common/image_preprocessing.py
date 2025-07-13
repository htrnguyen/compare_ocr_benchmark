import cv2
import numpy as np
from PIL import Image, ImageEnhance


def is_blurry(image_pil, threshold=100.0):
    """
    Kiểm tra ảnh có bị mờ không bằng biến đổi Laplacian.
    """
    image = np.array(image_pil.convert("L"))  # chuyển grayscale
    fm = cv2.Laplacian(image, cv2.CV_64F).var()
    return fm < threshold, fm


def auto_contrast(image_pil):
    """
    Tăng tương phản tự động bằng PIL.
    """
    return ImageEnhance.Contrast(image_pil).enhance(1.5)


def auto_sharpen(image_pil):
    """
    Làm sắc nét ảnh.
    """
    return ImageEnhance.Sharpness(image_pil).enhance(2.0)


def auto_brightness(image_pil):
    """
    Tăng độ sáng ảnh.
    """
    return ImageEnhance.Brightness(image_pil).enhance(1.2)


def resize_with_padding(image_pil, target_size=448):
    """
    Resize ảnh về target_size, giữ tỷ lệ, thêm padding nếu cần.
    """
    old_size = image_pil.size  # (width, height)
    ratio = float(target_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    image = image_pil.resize(new_size, Image.BICUBIC)
    new_im = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    new_im.paste(
        image, ((target_size - new_size[0]) // 2, (target_size - new_size[1]) // 2)
    )
    return new_im


def preprocess_image(image_path, target_size=448, do_auto=True):
    """
    Quy trình tiền xử lý ảnh tổng quát, áp dụng cho mọi model.
    - Resize, padding về đúng kích thước.
    - Tăng tương phản, sáng, làm nét nếu ảnh mờ hoặc nhạt.
    - Trả về cả ảnh PIL đã xử lý và mảng numpy.
    """
    img = Image.open(image_path).convert("RGB")
    # Resize và padding
    img = resize_with_padding(img, target_size)

    if do_auto:
        # Kiểm tra mờ tự động
        blurry, score = is_blurry(img)
        if blurry:
            img = auto_sharpen(img)
        # Tự tăng sáng/tương phản nếu độ tương phản thấp
        img = auto_contrast(img)
        img = auto_brightness(img)

    img_np = np.array(img)
    return img, img_np
