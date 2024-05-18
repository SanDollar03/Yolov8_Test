import cv2
import torch
from ultralytics import YOLO

# デバイスの設定
device = torch.device('cpu')

# YOLOv8xモデルをロード
model = YOLO('yolov8x.pt')
model.to(device)  # モデルをCPUに移動

# カメラを初期化
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8で推論を実行
    results = model(frame)

    # 結果をフレームに描画
    annotated_frame = results[0].plot()

    # フレームを表示
    cv2.imshow('YOLOv8 Real-Time Detection', annotated_frame)

    # 'q'キーを押すとループを終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# カメラとウィンドウを解放
cap.release()
cv2.destroyAllWindows()
