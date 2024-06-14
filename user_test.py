import requests
import base64
from PIL import Image
from io import BytesIO
import json
import os

# API 엔드포인트 URL
url = 'http://localhost:5000/brain'

# GET 요청 보내기
response = requests.get(url)

# 응답 확인
data = response.json()

# 결과를 저장할 디렉터리 생성
os.makedirs('result', exist_ok=True)

"""for user_id, user_info in data.items():
    if 'image' in user_info:    
        # base64 이미지 디코딩 및 저장
        image_data = base64.b64decode(user_info['image'])
        image = Image.open(BytesIO(image_data))
        image_path = f"result/{user_id}.jpg"
        image.save(image_path)

        # JSON에서 image 필드 제거
        del user_info['image']"""

# JSON 형식으로 출력
print(json.dumps(data, ensure_ascii=False, indent=4))

