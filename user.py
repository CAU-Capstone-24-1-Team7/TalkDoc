import os
import base64
from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)

# 가짜 사용자 데이터 폴더 경로
user_data_folder = 'user_data'
user_images_folder = 'user_images'

# API 엔드포인트: 사용자 정보 가져오기
@app.route('/api/user/<int:user_id>', methods=['GET'])
def get_user_info(user_id):
    user_info_file = os.path.join(user_data_folder, f"{user_id}.txt")
    try:
        with open(user_info_file, 'r') as f:
            user_info = f.readlines()
        user_info = [line.strip() for line in user_info]
        image_file = os.path.join(user_images_folder, f"{user_id}.jpg")
        if os.path.exists(image_file):
            with open(image_file, 'rb') as f:
                image_data = f.read()
            # 이미지를 base64로 인코딩
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            return jsonify({"Code": "200", "name": user_info[0].split(': ')[1], "address": user_info[1].split(': ')[1], "email": user_info[2].split(': ')[1], "birthdate": user_info[3].split(': ')[1], "phone_number": user_info[4].split(': ')[1], "image": image_base64})
        else:
            return jsonify({"Code": "404", "error": "Image not found"}), 404
    except FileNotFoundError:
        return jsonify({"Code": "404", "error": "User not found"}), 404

# GET 요청에 대한 이미지와 정보 반환
@app.route('/api/user', methods=['GET'])
def get_users():
    user_id = request.args.get('id')
    if user_id:
        return get_user_info(user_id)
    else:
        all_users_info = {}
        for user_id in range(1, 20):  # 1부터 20까지의 사용자 ID에 대해 반복
            user_info_file = os.path.join(user_data_folder, f"{user_id}.txt")
            try:
                with open(user_info_file, 'r') as f:
                    user_info = f.readlines()
                user_info = [line.strip() for line in user_info]
                image_file = os.path.join(user_images_folder, f"{user_id}.jpg")
                if os.path.exists(image_file):
                    with open(image_file, 'rb') as f:
                        image_data = f.read()
                    # 이미지를 base64로 인코딩
                    print(user_id,user_info[0].split(': ')[1])
                    print(user_info[1].split(': ')[1])
                    print(user_info[2])
                    print(user_info[3].split(': ')[1])
                    print(user_info[4].split(': ')[1])
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                    all_users_info[user_id] = {"name": user_info[0].split(': ')[1], "address": user_info[1].split(': ')[1], "email": user_info[2].split(': ')[1], "birthdate": user_info[3].split(': ')[1], "phone_number": user_info[4].split(': ')[1], "image": image_base64}
                else:
                    all_users_info[user_id] = {"name": user_info[0].split(': ')[1], "address": user_info[1].split(': ')[1], "email": user_info[2].split(': ')[1], "birthdate": user_info[3].split(': ')[1], "phone_number": user_info[4].split(': ')[1], "image": ""}
            except FileNotFoundError:
                all_users_info[user_id] = {"error": "User not found"}
        print(all_users_info)
        return jsonify(all_users_info)

# aqs.txt 파일 경로
aqs_file_path = './brain/aqs.txt'

# API 엔드포인트: aqs.txt 내용 가져오기
@app.route('/brain', methods=['GET'])
def get_aqs():
    try:
        with open(aqs_file_path, 'r', encoding='utf-8') as f:
            aqs_data = f.read().strip()
        return jsonify({"Code": 200, "data": aqs_data})
    except FileNotFoundError:
        return jsonify({"Code": 404, "error": "File not found"}), 404
    except UnicodeDecodeError as e:
        return jsonify({"Code": 500, "error": f"Encoding error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)


if __name__ == '__main__':
    app.run(debug=True)
