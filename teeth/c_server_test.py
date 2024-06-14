import requests
import base64

# 이미지를 base64로 인코딩
with open("teeth_dataset/cavity_1.jpg", "rb") as img_file:
    b64_string = base64.b64encode(img_file.read()).decode('utf-8')

# JSON 데이터 준비
data = {'image': b64_string}

# 서버로 요청 보내기
response = requests.post("http://127.0.0.1:5000/predict", json=data)

# 응답 출력
print(response.json())
