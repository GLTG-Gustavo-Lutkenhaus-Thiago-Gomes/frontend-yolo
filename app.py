from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
import cv2
import numpy as np
import base64
from ultralytics import YOLO
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY')
socketio = SocketIO(app, cors_allowed_origins="*")

model = YOLO('./model/yolov8n.pt')

latest_detections = []

@socketio.on('frame')
def handle_frame(data):
    if not data or ',' not in data:
        print("Frame recebido está vazio ou inválido. Dado recebido:", data)
        return
    
    encoded_data = data.split(',')[1]
    if not encoded_data.strip():
        print("Dados base64 estão vazios.")
        return

    try:
        frame = np.frombuffer(base64.b64decode(encoded_data), dtype=np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        if frame is None:
            print("O OpenCV não conseguiu decodificar o frame.")
            return

        results = model(frame, imgsz=640, conf=0.85)
        annotated_frame = results[0].plot()

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        processed_frame = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('processed_frame', processed_frame)
    except Exception as e:
        print(f"Erro ao processar frame: {e}")

@app.route('/detections')
def get_detections():
    global latest_detections
    return jsonify(latest_detections)

@app.route('/deteccao.html')
def get_deteccao():
    return render_template('deteccao.html')

@app.route('/sobre.html')
def get_sobre():
    return render_template('sobre.html')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
