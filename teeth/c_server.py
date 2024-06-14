from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# 모델 로드
loaded_model = load_model("model.h5")

@app.route('/predict', methods=['POST'])
def predict():
    # 요청에서 base64로 인코딩된 이미지 데이터 받기
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    
    # base64 디코딩
    img_data = base64.b64decode(data['image'])
    img = Image.open(BytesIO(img_data))
    img = img.resize((64, 64))  # 이미지 크기 조정

    # 이미지 전처리
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # 모델 예측
    predictions = loaded_model.predict(img_array)
    prediction = predictions[0, 0]

    # 결과 반환
    result = {
        'prediction': float(prediction),
        'result': 'Cavity' if prediction < 0.5 else 'No Cavity'
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
