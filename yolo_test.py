import torch
from ultralytics import YOLO

# デバイスの設定
device = torch.device('cpu')

# YOLOv8n (Nano) モデルをロード
model = YOLO('yolov8n.pt')
model.to(device)  # モデルをCPUに移動

# 画像を使用して予測を実行
results = model('C:/Users/hardm/Downloads/image.jpg')  # 画像のパスを指定

# 予測結果を表示
for result in results:
    result.show()
