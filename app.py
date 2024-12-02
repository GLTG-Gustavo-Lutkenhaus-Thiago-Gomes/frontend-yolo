from flask import Flask, Response, render_template, request, redirect, url_for, jsonify
import cv2
from ultralytics import YOLO
from time import sleep, time

app = Flask(__name__)

model = YOLO('./model/yolov8n.pt')

cv2.setNumThreads(1)
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

latest_detections = []
MAX_FPS = 3
frame_interval = 1.0 / MAX_FPS

def generate_frames():
    global latest_detections
    last_time = time()

    while True:
        if not camera.isOpened():
            print("Erro: A câmera não pode ser acessada. Verifique o índice ou a conexão.")

        success, frame = camera.read()
        if not success:
            break

        current_time = time()
        if current_time - last_time < frame_interval:
            sleep(frame_interval - (current_time - last_time))
            continue
        last_time = current_time

        results = model(frame, imgsz=640, conf=0.7)
        detections = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0]) if box.cls is not None else None
                conf = float(box.conf[0]) if box.conf is not None else None
                if cls is not None and conf is not None:
                    detections.append({
                        "class": model.names[cls],
                        "confidence": round(conf, 2)
                    })
            annotated_frame = result.plot()

        latest_detections = detections

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    sleep(1)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def get_detections():
    global latest_detections
    return jsonify(latest_detections)

@app.route('/deteccao.html')
def get_deteccao():
    return render_template('deteccao.html')

@app.route('/contato.html')
def get_contato():
    return render_template('contato.html')

@app.route('/sobre.html')
def get_sobre():
    return render_template('sobre.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_resolution', methods=['POST'])
def set_resolution():
    width = int(request.form.get('width', 640))
    height = int(request.form.get('height', 480))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return redirect(url_for('get_deteccao'))

if __name__ == '__main__':
    app.run(debug=True)
