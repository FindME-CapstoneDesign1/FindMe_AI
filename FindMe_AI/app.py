from flask import Flask, request, jsonify
import torch
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import logging
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 라벨 매핑 딕셔너리
label_mapping = {
    'bag': '가방', 'jewelry': '귀금속', 'others': '기타물품', 'books': '도서용품',
    'documents': '서류', 'shoppingbag': '쇼핑백', 'sportsEquipment': '스포츠용품',
    'instrument': '악기', 'securities': '유가증권', 'clothing': '의류', 'car': '자동차',
    'electronics': '전자기기', 'certificate': '증명서', 'wallet': '지갑', 'card': '카드',
    'computer': '컴퓨터', 'cash': '현금', 'phone': '휴대폰'
}

def load_image_from_bytes(image_bytes):
    try:
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        return np.array(image)
    except Exception as e:
        logging.error(f"Error loading the image from bytes: {e}")
        return None

def resize_image(image, target_size=(640, 640)):
    # 이미지를 YOLO 모델에 맞게 크기 조정
    return cv2.resize(image, target_size)

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        image_bytes = file.read()

        if not image_bytes:
            return jsonify({'error': 'Empty image file'}), 400

        print(f"Received image file: {file.filename}")
        
        # YOLOv5 모델 로드
        model_path = 'models/best.pt'
        try:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        except Exception as e:
            logging.error(f"Model loading failed: {e}")
            return jsonify({'error': f"Model loading failed: {e}"}), 500

        # 이미지 바이트에서 이미지 로드
        input_image = load_image_from_bytes(image_bytes)

        # input_image가 None이거나 빈 배열인지 확인
        if input_image is None or input_image.size == 0:
            return jsonify({'error': 'Failed to load the input image or image is empty'}), 400

        # 이미지 크기 조정
        resized_image = resize_image(input_image)

        # YOLOv5 모델로 이미지 분석
        results = model(resized_image)

        # 결과 추출
        df = results.pandas().xyxy[0]
        if df.empty:
            return jsonify({'error': 'No objects detected in the image'}), 400

        df['area'] = (df['xmax'] - df['xmin']) * (df['ymax'] - df['ymin'])
        largest_label = df.loc[df['area'].idxmax()]['name']
        translated_label = label_mapping.get(largest_label, largest_label)

        response_data = {'translated_label': translated_label}
        response = json.dumps(response_data, ensure_ascii=False)
        return response, 200

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({'error': f"An error occurred: {e}"}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=True)
