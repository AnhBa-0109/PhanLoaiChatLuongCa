"""TunaNet web server. Run: python app.py --model outputs/runs/<name>/best.pt"""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import re
import threading
import time
import warnings

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from werkzeug.exceptions import RequestEntityTooLarge

from scripts.predict import load_model, predict_one

ROOT = Path(__file__).resolve().parent
MAX_FILE = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
METHODS = {'threshold': 'Ngưỡng màu', 'yolo': 'YOLO-seg'}
DEFAULT_MODEL = 'outputs/runs/cv_s2/best.pt'            # TunaNet học trên ảnh luồng ngưỡng màu
DEFAULT_YOLO_MODEL = 'outputs/runs/yolo_s2/best.pt'     # TunaNet học trên ảnh luồng YOLO-seg
DEFAULT_SEG_MODEL = 'runs/segment/outputs/yolo/seg_meat1/weights/best.pt'


def load_run(name):
    try:
        data = json.loads((ROOT / 'outputs' / 'runs' / name / 'results.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return {'name': name, 'accuracy': data['test_acc'], 'macro_f1': data['test_f1_macro'],
            'f1': data['test_f1_per_class'], 'confusion': data['confusion'], 'classes': data['classes'],
            'best_epoch': data.get('best_epoch')}


def mean_std(values):
    values = np.array(values, dtype=float)
    return {'mean': float(values.mean()), 'std': float(values.std(ddof=1)) if len(values) > 1 else 0.0}


def summarize(label, names):
    runs = [run for run in map(load_run, names) if run]
    if not runs:
        return {'label': label, 'runs': []}
    return {'label': label, 'runs': runs, 'accuracy': mean_std([r['accuracy'] for r in runs]),
            'macro_f1': mean_std([r['macro_f1'] for r in runs]),
            'f1': {c: mean_std([r['f1'][c] for r in runs]) for c in runs[0]['classes']},
            'best': max(runs, key=lambda r: r['accuracy'])}


def read_iou():
    """Đọc bảng IoU trong report.txt do preprocess_yolo.py tạo ra."""
    try:
        text = (ROOT / 'data' / 'processed_yolo' / 'report.txt').read_text(encoding='utf-8')
    except OSError:
        return []
    rows = re.findall(r'^\s*(train|val|test)\s+(\d+)\s+([\d.]+)\s+([\d.]+).*?IoU < 0\.8: \S+ (\d+) ảnh, ngưỡng màu (\d+) ảnh',
                      text, re.MULTILINE)
    return [{'split': s, 'images': int(n), 'yolo': float(y), 'threshold': float(c),
             'yolo_low': int(yl), 'threshold_low': int(cl)} for s, n, y, c, yl, cl in rows]


def experiment_data():
    return {'pipelines': [summarize('Không dùng YOLO (ngưỡng màu)', ['cv_s1', 'cv_s2', 'cv_s3']),
                          summarize('Dùng YOLO-seg', ['yolo_s1', 'yolo_s2', 'yolo_s3'])],
            'white_balance': [run for run in (load_run('wb_off'), load_run('wb_on')) if run],
            'iou': read_iou()}


def to_data_url(img, side=1000):
    h, w = img.shape[:2]
    scale = side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError('Cannot encode preview')
    return 'data:image/jpeg;base64,' + base64.b64encode(encoded).decode('ascii')


def outlined(bgr, mask, color=(220, 220, 40)):
    out = bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, max(2, max(bgr.shape[:2]) // 300))
    return out


def channel_a(bgr, mask):
    """Kênh a của Lab, kéo giãn tương phản trong vùng thịt để dễ quan sát."""
    a = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 1].astype(np.float32)
    values = a[mask > 0] if (mask > 0).any() else a
    low, high = np.percentile(values, [1, 99])
    return np.clip((a - low) / max(high - low, 1) * 255, 0, 255).astype(np.uint8)


def stage_list(items):
    return [{'label': label, 'image': to_data_url(img)} for label, img in items]


def build_result(prob, classes, crop, mask, ratio, box, warning=None, stages=None, timings=None):
    if not np.isfinite(prob).all():
        raise ValueError('Model returned non-finite probabilities')
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = crop.copy()
    cv2.drawContours(overlay, contours, -1, (220, 220, 40), 2)
    ok, encoded = cv2.imencode('.jpg', overlay)
    if not ok:
        raise ValueError('Cannot encode preview')
    k = int(prob.argmax())
    if ratio < 0.15 or ratio > 0.97:
        warning = warning or 'Vùng thịt nhận diện bất thường. Hãy kiểm tra ảnh và vùng viền được đánh dấu.'
    return {'prediction': classes[k], 'confidence': float(prob[k]),
            'probabilities': [{'label': c, 'probability': float(p)} for c, p in zip(classes, prob)],
            'meat_ratio': ratio, 'box': box,
            'processed_image': 'data:image/jpeg;base64,' + base64.b64encode(encoded).decode('ascii'),
            'warning': warning, 'stages': stages or [], 'timings': timings or []}


class Predictor:
    def __init__(self, model_path):
        self.path = Path(model_path)
        if not self.path.is_absolute():
            self.path = ROOT / self.path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.classes = []
        self.error = None
        self.lock = threading.Lock()
        if self.path.is_file():
            try:
                self.model, self.classes, self.size = load_model(self.path, self.device)
            except Exception:
                self.error = 'Không thể nạp checkpoint. Kiểm tra file trọng số và khởi động lại server.'
        else:
            self.error = 'Chưa có trọng số mô hình. Thêm best.pt rồi khởi động server với --model đường_dẫn/best.pt.'

    def status(self):
        return {'ready': self.model is not None, 'message': self.error,
                'device': self.device.upper(), 'classes': self.classes,
                'model': self.path.name}

    def predict(self, bgr):
        with self.lock:
            prob, box, crop, mask, ratio = predict_one(self.model, bgr, self.size, self.device)
        if box is None:
            before = bgr
        else:
            x1, y1, x2, y2 = box
            before = bgr[y1:y2, x1:x2]
        stages = stage_list([('Ảnh gốc', bgr), ('Cắt quanh miếng thịt', before), ('Cân bằng trắng', crop),
                             ('Mask ngưỡng màu', mask), ('Kênh a của ảnh Lab', channel_a(crop, mask)),
                             ('Vùng thịt đưa vào TunaNet', outlined(crop, mask))])
        return build_result(prob, self.classes, crop, mask, ratio, box, stages=stages)


class YoloPredictor:
    """Luồng YOLO-seg -> OpenCV -> TunaNet, dùng đúng run_pipeline của predict_full.py."""

    def __init__(self, tunanet_path, seg_path):
        self.tunanet_path = Path(tunanet_path) if Path(tunanet_path).is_absolute() else ROOT / tunanet_path
        self.seg_path = Path(seg_path) if Path(seg_path).is_absolute() else ROOT / seg_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.classes = []
        self.error = None
        self.lock = threading.Lock()
        missing = [p for p in (self.tunanet_path, self.seg_path) if not p.is_file()]
        if missing:
            self.error = 'Chưa có trọng số cho YOLO-seg: ' + ', '.join(p.as_posix() for p in missing)
            return
        try:
            from ultralytics import YOLO
            from scripts.predict_full import load_tunanet, run_pipeline
            self.run_pipeline = run_pipeline
            self.seg = YOLO(str(self.seg_path))
            self.model, self.classes, self.size = load_tunanet(self.tunanet_path, self.device)
        except Exception:
            self.model = None
            self.error = 'Không thể nạp YOLO-seg. Kiểm tra thư viện ultralytics và file trọng số.'

    def status(self):
        return {'ready': self.model is not None, 'message': self.error,
                'device': self.device.upper(), 'classes': self.classes,
                'model': self.tunanet_path.name}

    def predict(self, bgr):
        with self.lock:
            res = self.run_pipeline(bgr, self.seg, self.model, self.size, self.device)
        mask = res['mcrop']
        ratio = float((mask > 0).mean())
        warning = None
        if res['mask_src'] != 'yolo':
            warning = 'YOLO-seg không tìm thấy miếng thịt nên hệ thống đã dùng ngưỡng màu để tạo mask.'
        box = [int(v) for v in res['box']]
        x1, y1, x2, y2 = box
        mask_label = 'Mask YOLO-seg' if res['mask_src'] == 'yolo' else 'Mask dự phòng (ngưỡng màu)'
        stages = stage_list([('Ảnh gốc', bgr), (mask_label, outlined(bgr, res['mask_full'])),
                             ('Cắt rộng (lề 15%)', bgr[y1:y2, x1:x2]), ('Cân bằng trắng', res['wide_wb']),
                             ('Cắt sát (lề 5%)', res['crop']), ('Kênh a của ảnh Lab', channel_a(res['crop'], mask)),
                             ('Mask đưa vào TunaNet', mask)])
        names = {'yolo': 'YOLO-seg', 'opencv': 'OpenCV', 'tunanet': 'TunaNet'}
        timings = [{'label': names[k], 'ms': round(v * 1000)} for k, v in res['time'].items() if k in names]
        return build_result(res['prob'], self.classes, res['crop'], mask, ratio, box, warning, stages, timings)


def decode_image(raw):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as im:
                if im.format not in {'JPEG', 'PNG', 'WEBP', 'BMP'}:
                    raise ValueError('Chỉ hỗ trợ ảnh JPG, PNG, WEBP hoặc BMP.')
                if im.width * im.height > MAX_PIXELS or min(im.size) < 32 or max(im.size) / min(im.size) > 20:
                    raise ValueError('Ảnh phải có mỗi cạnh ít nhất 32 px, tối đa 20 megapixel và tỉ lệ cạnh không quá 20:1.')
                im.load()
                rgb = np.array(ImageOps.exif_transpose(im).convert('RGB'))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError('Không đọc được ảnh. Hãy chọn một file ảnh hợp lệ, không bị hỏng.') from exc


def create_app(model_path=None, predictor=None, yolo_predictor=None):
    app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
    # Waitress does not enable Flask's debug template reloader by default.
    # Recheck HTML on each request so a running server cannot mix old HTML
    # with freshly edited CSS/JS.
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['MAX_CONTENT_LENGTH'] = MAX_FILE + 1024 * 1024
    service = predictor if predictor is not None else Predictor(
        model_path or os.environ.get('TUNANET_MODEL', DEFAULT_MODEL))
    app.extensions['predictor'] = service
    app.extensions['yolo_predictor'] = yolo_predictor
    services = {'threshold': service, 'yolo': yolo_predictor}

    def method_status(name):
        if services[name] is None:
            return {'label': METHODS[name], 'ready': False, 'message': 'Server chưa bật YOLO-seg.'}
        state = services[name].status()
        return {'label': METHODS[name], 'ready': state['ready'], 'message': state['message']}

    @app.get('/')
    def index():
        versions = {name: (Path(app.static_folder) / name).stat().st_mtime_ns
                    for name in ('style.css', 'app.js')}
        return render_template('index.html', asset_versions=versions)

    @app.get('/api/health')
    def health():
        return jsonify({**service.status(), 'methods': {name: method_status(name) for name in METHODS}})

    @app.get('/api/experiments')
    def experiments():
        return jsonify(experiment_data())

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):
        return jsonify(error='Ảnh quá lớn. Dung lượng tối đa là 10 MB.'), 413

    @app.post('/api/predict')
    def predict():
        method = request.form.get('method', 'threshold')
        if method not in METHODS:
            return jsonify(error='Phương án tách vùng thịt không hợp lệ.'), 400
        upload = request.files.get('image')
        if upload is None or not upload.filename:
            return jsonify(error='Vui lòng chọn một ảnh.'), 400
        raw = upload.read(MAX_FILE + 1)
        if len(raw) > MAX_FILE:
            return too_large(None)
        try:
            bgr = decode_image(raw)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        state = method_status(method)
        if not state['ready']:
            return jsonify(error=state['message']), 503
        start = time.perf_counter()
        try:
            result = services[method].predict(bgr)
        except Exception:
            app.logger.exception('TunaNet inference failed')
            return jsonify(error='Không thể xử lý ảnh. Hãy thử ảnh khác hoặc kiểm tra log server.'), 500
        result['elapsed_ms'] = round((time.perf_counter() - start) * 1000)
        result['method'], result['method_label'] = method, METHODS[method]
        return jsonify(result)

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Cache-Control'] = 'no-store'
        return response

    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TunaNet web application')
    parser.add_argument('--model', default=os.environ.get('TUNANET_MODEL', DEFAULT_MODEL),
                        help='TunaNet cho luồng ngưỡng màu')
    parser.add_argument('--yolo-model', default=DEFAULT_YOLO_MODEL, help='TunaNet cho luồng YOLO-seg')
    parser.add_argument('--seg-model', default=DEFAULT_SEG_MODEL, help='trọng số YOLO-seg')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    torch.set_num_threads(8)
    from waitress import serve
    application = create_app(args.model, yolo_predictor=YoloPredictor(args.yolo_model, args.seg_model))
    print(f'TunaNet: http://{args.host}:{args.port}', flush=True)
    state = application.extensions['predictor'].status()
    yolo_state = application.extensions['yolo_predictor'].status()
    print(f"Ngưỡng màu ready: {state['ready']} | YOLO-seg ready: {yolo_state['ready']} | Device: {state['device']}",
          flush=True)
    if yolo_state['message']:
        print(yolo_state['message'], flush=True)
    serve(application, host=args.host, port=args.port, threads=4,
          max_request_body_size=MAX_FILE + 2 * 1024 * 1024)
