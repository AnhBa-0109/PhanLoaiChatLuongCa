# =====================================================================
#  CHẠY TRỌN LUỒNG CHÍNH:  YOLO-seg -> OpenCV -> TunaNet -> A/B/C
#
#  Cách chạy (đứng ở thư mục TunaNet, đã bật .venv):
#     .\run_yolo.ps1 -RF "D:\StudyUni\AI\tuna_roboflow"
#  Đã train YOLO rồi, chỉ muốn chạy lại phần sau:
#     .\run_yolo.ps1 -RF "D:\StudyUni\AI\tuna_roboflow" -SkipYolo
#  Nếu PowerShell báo bị chặn chạy script, chạy trước 1 lần:
#     Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# =====================================================================
param(
    [Parameter(Mandatory = $true)][string]$RF,   # thư mục giải nén Roboflow
    [int]$YoloEpochs = 50,
    [int]$Epochs = 30,
    [int]$Seed = 1,
    [switch]$SkipYolo
)

Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Name = "yolo_s$Seed"

function Step($title, [scriptblock]$cmd) {
    Write-Host "`n==================== $title ====================" -ForegroundColor Cyan
    $t = Get-Date
    & $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[LỖI] Bước '$title' thất bại. Dừng lại." -ForegroundColor Red
        exit 1
    }
    Write-Host ("[XONG] {0} ({1:N0} giây)" -f $title, ((Get-Date) - $t).TotalSeconds) -ForegroundColor Green
}

if (-not (Test-Path $RF)) { Write-Host "Không thấy thư mục Roboflow: $RF" -ForegroundColor Red; exit 1 }

Step "0. Kiểm tra môi trường"   { python check_env.py }
Step "1. Cắt ảnh từ Roboflow"   { python scripts/roboflow_to_crops.py --src $RF }
Step "2. Chia tập theo buổi"    { python scripts/make_splits.py }
Step "3. Tạo dữ liệu YOLO-seg"  { python scripts/prepare_yolo.py --src $RF --task seg --classes 1 --preview 6 }

if (-not $SkipYolo) {
    Step "4. Train YOLO-seg" {
        yolo segment train data=data/yolo/seg_meat1_split/data.yaml model=yolov8n-seg.pt imgsz=640 `
            epochs=$YoloEpochs batch=16 device=cpu workers=0 project=outputs/yolo name=seg_meat1 exist_ok=True
    }
}

# Ultralytics có bản lưu vào runs/segment/outputs/yolo/..., có bản lưu vào outputs/yolo/...
$YoloPt = @("runs/segment/outputs/yolo/seg_meat1/weights/best.pt",
            "outputs/yolo/seg_meat1/weights/best.pt") | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $YoloPt) { Write-Host "Không tìm thấy best.pt của YOLO-seg" -ForegroundColor Red; exit 1 }
Write-Host "Dùng YOLO-seg: $YoloPt"

Step "5. YOLO + OpenCV xử lý ảnh" { python scripts/preprocess_yolo.py --src $RF --model $YoloPt }
Step "6. Train TunaNet" {
    python scripts/train.py --processed data/processed_yolo/processed.csv --epochs $Epochs --image-size 256 --seed $Seed --name $Name
}
if (Test-Path "test_image") {
    Step "7. Dự đoán ảnh thử" {
        python scripts/predict_full.py --input test_image --yolo $YoloPt --model "outputs/runs/$Name/best.pt"
    }
}

Write-Host "`nHOÀN TẤT. Xem kết quả:" -ForegroundColor Green
Write-Host "  - Mask YOLO, IoU      : data/processed_yolo/report.txt"
Write-Host "  - Kết quả TunaNet     : outputs/runs/$Name/results.json, confusion.png"
Write-Host "  - Ảnh dự đoán         : outputs/predictions_full/"