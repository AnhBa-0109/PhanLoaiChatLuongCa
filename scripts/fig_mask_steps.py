"""
Vẽ Hình 3.1: các bước tạo mask bằng ngưỡng màu trên MỘT ảnh.
Ảnh gốc | Ảnh điểm | Sau Otsu | Mask cuối

Đặt file này vào TunaNet/scripts/, rồi chạy từ thư mục TunaNet:
    python scripts/fig_mask_steps.py                     # tự lấy một ảnh trong data/roboflow_crops
    python scripts/fig_mask_steps.py --image duong_dan_anh.jpg

Kết quả: outputs/figures/hinh_3_1_mask_steps.png
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preprocess import fill_holes, make_mask, resize_max  # noqa: E402


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def steps(bgr):
    """Các bước giống make_mask, nhưng giữ lại kết quả trung gian."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    score = 1.5 * np.clip(a, 0, None) + 0.8 * np.clip(b, 0, None) + 0.5 * np.clip(255 - L, 0, None)
    score = cv2.GaussianBlur(score, (0, 0), 2)
    score = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    t, otsu = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = max(3, (min(otsu.shape) // 60) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, ker)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker, iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        m = np.where(labels == big, 255, 0).astype(np.uint8)
    return score, otsu, fill_holes(m), t


def pick_image():
    p = Path("data/roboflow_crops/crops.csv")
    if not p.exists():
        sys.exit("Không thấy data/roboflow_crops/crops.csv. Hãy dùng --image <đường dẫn ảnh>.")
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    return p.parent / rows[len(rows) // 2]["file"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--out", default="outputs/figures/hinh_3_1_mask_steps.png")
    args = ap.parse_args()

    path = Path(args.image) if args.image else pick_image()
    bgr = imread(path)
    if bgr is None:
        sys.exit(f"Không đọc được ảnh: {path}")
    bgr = resize_max(bgr, args.max_side)
    score, otsu, final, t = steps(bgr)

    # Kiểm tra: ảnh trung gian phải khớp với hàm make_mask thật của dự án
    real = make_mask(bgr)
    same = float(((real > 0) == (final > 0)).mean())
    print(f"Ảnh: {path}")
    print(f"Ngưỡng Otsu: {t:.0f}")
    print(f"Độ trùng với make_mask: {same:.2%}")
    if same < 0.999:
        print("[CẢNH BÁO] Script không khớp make_mask. make_mask trên máy đã khác, gửi lại file để sửa.")

    titles = ["(a) Ảnh gốc", "(b) Ảnh điểm", f"(c) Sau Otsu (ngưỡng {t:.0f})", "(d) Mask cuối"]
    imgs = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), score, otsu, final]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for ax, im, ti in zip(axes, imgs, titles):
        ax.imshow(im, cmap=None if im.ndim == 3 else "gray", vmin=0, vmax=255)
        ax.set_title(ti, fontsize=12)
        ax.axis("off")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Đã lưu: {out.resolve()}")


if __name__ == "__main__":
    main()