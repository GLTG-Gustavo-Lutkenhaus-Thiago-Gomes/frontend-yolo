from flask import Flask, Response, render_template, request, redirect, url_for, jsonify
import cv2
from ultralytics import YOLO
from time import sleep


app = Flask(__name__)

# Carrega o modelo YOLOv8n treinado
model = YOLO("./model/yolov8n.pt")

# Configuração inicial
camera = cv2.VideoCapture(0)  # Usa a câmera padrão do sistema
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Variável global para armazenar detecções
latest_detections = []

# Função para capturar a câmera e aplicar o YOLOv8n
def generate_frames():
    global latest_detections
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Realiza a detecção
            results = model(frame, imgsz=640)
            detections = []
            for result in results:
                for box in result.boxes:
                    # Extraia informações como classe e confiança
                    cls = int(box.cls[0]) if box.cls is not None else None
                    conf = float(box.conf[0]) if box.conf is not None else None
                    if cls is not None and conf is not None:
                        detections.append({
                            "class": model.names[cls],  # Nome da classe
                            "confidence": round(conf, 2)  # Confiança arredondada
                        })
                # Desenha as caixas delimitadoras no frame
                annotated_frame = result.plot()

            # Atualiza a variável global com as detecções atuais
            latest_detections = detections

            # Converte frame para bytes
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Rota para o feed de vídeo
@app.route('/video_feed')
def video_feed():
    sleep(1)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Rota para fornecer as detecções como JSON
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

# Página inicial
@app.route('/')
def index():
    return render_template('index.html')

# Rota para ajustar resolução
@app.route('/set_resolution', methods=['POST'])
def set_resolution():
    width = int(request.form.get('width', 640))
    height = int(request.form.get('height', 480))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return redirect(url_for('get_deteccao'))

if __name__ == '__main__':
    app.run(debug=True)
