"""
Vẽ các hình cho báo cáo từ MỘT ảnh gốc, dùng đúng code của dự án:
    Hình 2.3  Ảnh mẫu ba cấp A, B, C
    Hình 3.2  Mask của YOLO-seg
    Hình 3.3  Ảnh qua các bước chuẩn hóa
    Hình 3.4  Trước và sau cân bằng trắng
    Hình 3.5  Ba kênh L, a, b

Đặt file này vào TunaNet/scripts/, rồi chạy từ thư mục TunaNet:
    python scripts/fig_chuong3.py --image test_image\\anh1.jpg

Kết quả: outputs/figures/hinh_*.png
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from preprocess_yolo import crop_box, yolo_mask  # noqa: E402
from src.data.preprocess import resize_max, white_balance  # noqa: E402

YOLO_PATHS = ["runs/segment/outputs/yolo/seg_meat1/weights/best.pt", "outputs/yolo/seg_meat1/weights/best.pt"]


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def show(panels, out, name):
    """panels = [(ảnh, tiêu đề), ...]"""
    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.4))
    for ax, (im, title) in zip(np.atleast_1d(axes), panels):
        if im.ndim == 3:
            ax.imshow(im)
        else:
            ax.imshow(im, cmap="gray")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    path = out / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã lưu: {path.resolve()}")


def overlay(bgr, mask):
    vis = bgr.copy()
    m = mask > 0
    vis[m] = (0.6 * vis[m] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 255), max(2, bgr.shape[1] // 250))
    return vis


def samples(out):
    p = Path("data/roboflow_crops/crops.csv")
    if not p.exists():
        print("Bỏ qua Hình 2.3: không thấy data/roboflow_crops/crops.csv")
        return
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    panels = []
    for g in sorted({r["grade"] for r in rows}):
        files = [r["file"] for r in rows if r["grade"] == g]
        im = imread(p.parent / files[len(files) // 2])
        if im is not None:
            panels.append((rgb(im), g))
    show(panels, out, "hinh_2_3_mau_ba_cap.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="ảnh gốc, có nền giấy, ví dụ test_image\\anh1.jpg")
    ap.add_argument("--yolo", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--tight", type=float, default=0.05)
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--out", default="outputs/figures")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    samples(out)

    yolo_path = args.yolo or next((p for p in YOLO_PATHS if Path(p).exists()), None)
    if yolo_path is None:
        sys.exit("Không tìm thấy best.pt của YOLO-seg. Hãy dùng --yolo <đường dẫn>.")
    from ultralytics import YOLO

    bgr = imread(args.image)
    if bgr is None:
        sys.exit(f"Không đọc được ảnh: {args.image}")
    mask_full, conf = yolo_mask(YOLO(yolo_path), bgr, args.conf)
    if mask_full is None:
        sys.exit("YOLO-seg không tìm thấy miếng thịt trong ảnh này. Hãy chọn ảnh khác.")
    print(f"Độ tin cậy YOLO-seg: {conf:.3f}")

    # Hình 3.2
    show([(rgb(bgr), "(a) Ảnh gốc"), (rgb(overlay(bgr, mask_full)), "(b) Mask YOLO-seg")],
         out, "hinh_3_2_mask_yolo.png")

    # Các bước giống preprocess_yolo.py
    x1, y1, x2, y2 = crop_box(mask_full, args.margin)
    wide, mwide = bgr[y1:y2, x1:x2], mask_full[y1:y2, x1:x2]
    wide_wb, gains = white_balance(wide, mwide)
    print("Hệ số cân bằng trắng (B, G, R):", tuple(round(g, 3) for g in gains))
    tx1, ty1, tx2, ty2 = crop_box(mwide, args.tight)
    crop, mcrop = wide_wb[ty1:ty2, tx1:tx2], mwide[ty1:ty2, tx1:tx2]
    crop = resize_max(crop, args.max_side)
    mcrop = cv2.resize(mcrop, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)

    # Hình 3.3
    show([(rgb(wide), "(a) Cắt rộng"), (rgb(wide_wb), "(b) Cân bằng trắng"), (rgb(crop), "(c) Cắt sát"),
          (lab[..., 1], "(d) Kênh a của ảnh Lab"), (mcrop, "(e) Mask")], out, "hinh_3_3_chuan_hoa.png")
    # Hình 3.4
    show([(rgb(wide), "(a) Trước cân bằng trắng"), (rgb(wide_wb), "(b) Sau cân bằng trắng")],
         out, "hinh_3_4_can_bang_trang.png")
    # Hình 3.5: kéo giãn tương phản riêng từng kênh để dễ nhìn
    stretch = lambda c: cv2.normalize(c, None, 0, 255, cv2.NORM_MINMAX)
    show([(rgb(crop), "(a) Ảnh màu"), (lab[..., 0], "(b) Kênh L"), (stretch(lab[..., 1]), "(c) Kênh a"),
          (stretch(lab[..., 2]), "(d) Kênh b")], out, "hinh_3_5_kenh_lab.png")


if __name__ == "__main__":
    main()