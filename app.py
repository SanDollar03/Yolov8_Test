from flask import Flask, render_template, Response
import cv2
import torch
from ultralytics import YOLO

app = Flask(__name__)

# デバイスの設定
device = torch.device('cpu')

# YOLOv8nモデルをロード
model = YOLO('yolov8x.pt')
model.to(device)  # モデルをCPUに移動

# カメラを初期化
cap = cv2.VideoCapture(2)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # YOLOv8で推論を実行
            results = model(frame)

            # 結果をフレームに描画
            annotated_frame = results[0].plot()

            # フレームをJPEG形式にエンコード
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            # フレームをジェネレータとして返す
            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
