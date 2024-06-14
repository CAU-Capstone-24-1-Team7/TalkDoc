import requests
from io import BytesIO
import os
from PIL import Image
from faker import Faker

# Faker 객체 생성
fake = Faker('ko_KR')

# 폴더 생성
os.makedirs('user_images', exist_ok=True)
os.makedirs('user_data', exist_ok=True)

# URL에서 이미지를 가져오는 함수
def fetch_image(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content))
    return img

# 가짜 사용자 데이터 생성 및 저장
users = []
image_url = "https://thispersondoesnotexist.com"

for i in range(1, 21):
    # 사용자 데이터 생성
    user = {
        'name': fake.name(),
        'address': fake.address().replace('\n', '') ,
        'email': fake.email(),
        'birthdate': str(fake.date_of_birth()),
        'phone_number': fake.phone_number(),
    }
    
    # 사용자 이미지 다운로드 및 저장
    try:
        img = fetch_image(image_url)
        image_file = f"user_images/{i}.jpg"
        img.save(image_file)
        print(f"Success to download image for user {i}")
    except Exception as e:
        print(f"Failed to download image for user {i}: {e}")
        continue
    
    # 사용자 정보 저장
    info_file = f"user_data/{i}.txt"
    with open(info_file, 'w') as f:
        f.write(f"Name: {user['name']}\n")
        f.write(f"Address: {user['address']}\n")
        f.write(f"Email: {user['email']}\n")
        f.write(f"Birthdate: {user['birthdate']}\n")
        f.write(f"Phone Number: {user['phone_number']}\n")
        f.write(f"Image File: {image_file}\n")
        f.write(f"Brain Score: {"100"}\n")
    
    users.append(user)

# 결과 출력
for i, user in enumerate(users, start=1):
    print(f"User {i}: {user}")
