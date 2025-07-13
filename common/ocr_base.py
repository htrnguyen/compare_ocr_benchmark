class OCRModelBase:
    """
    Giao diện chuẩn cho mọi model OCR.
    """

    def __init__(self, config=None):
        self.config = config

    def predict(self, img_pil, img_np=None):
        """
        Nhận ảnh PIL (bắt buộc) và numpy (tùy model).
        Trả về text, thời gian nhận diện (giây).
        """
        raise NotImplementedError("Model chưa implement hàm predict!")

    def batch_predict(self, img_paths, preprocess_func):
        """
        Nhận danh sách đường dẫn ảnh, tự động preprocess, trả về danh sách kết quả.
        """
        results = []
        for img_path in img_paths:
            img_pil, img_np = preprocess_func(img_path)
            pred, t = self.predict(img_pil, img_np)
            results.append({"filename": img_path, "pred": pred, "time": t})
        return results
