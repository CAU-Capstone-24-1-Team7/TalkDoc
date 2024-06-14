import os
from speechbrain.pretrained import EncoderDecoderASR
import os
import base64
from flask import Flask, jsonify, request, Response
import subprocess
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

# Ensure the necessary directories exist
os.makedirs('user_data', exist_ok=True)
os.makedirs('login_data', exist_ok=True)
user_data_folder = 'user_data'
user_images_folder = 'user_images'

# 회원가입 엔드포인트
@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    auth = data.get('auth')
    name = data.get('name')
    ID = data.get('ID')
    password = data.get('password')
    patient_id = data.get('patient_id')
    print(auth)
    # ID 중복 확인
    if os.path.exists(f'login_data/{ID}.txt'):
        return '', 400
    if auth == '1':
        # 환자 파일 확인
        if os.path.exists(f'user_data/{patient_id}.txt'):
            print("login data created")
            # 로그인 데이터 파일 생성
            with open(f'login_data/{ID}.txt', 'w') as file:
                file.write(f'{password}\n{auth}\n{name}')
            return '', 200
        else:
            return '', 404
    elif auth in ['2', '3', '4']:
        # 환자 ID 확인 없이 로그인 데이터 파일 생성
        with open(f'login_data/{ID}.txt', 'w') as file:
            file.write(f'{password}\n{auth}\n{name}')
        return '', 200
    else:
        return '', 400

# 로그인 엔드포인트
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    ID = data.get('ID')
    password = data.get('password')

    # 로그인 데이터 파일 확인
    if not os.path.exists(f'login_data/{ID}.txt'):
        return '', 400

    # 로그인 데이터 파일 읽기
    with open(f'login_data/{ID}.txt', 'r') as file:
        stored_password = file.readline().strip()
        auth = file.readline().strip()
        name = file.readline().strip()

    if password == stored_password:
        return jsonify({'status_code': 200, 'auth': auth, 'name': name}), 200
    else:
        return '', 400

# 사용자 정보 가져오기 엔드포인트
@app.route('/api/user/<int:user_id>', methods=['GET'])
def get_user_info(user_id):
    user_info_file = os.path.join(user_data_folder, f"{user_id}.txt")
    try:
        with open(user_info_file, 'r', encoding='utf-8') as f:
            user_info = f.readlines()
        user_info_data = [line.strip() for line in user_info]
        image_file = os.path.join(user_images_folder, f"{user_id}.jpg")
        if os.path.exists(image_file):
            with open(image_file, 'rb') as f:
                image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            return Response(
                response=json.dumps(
                    {"Code": "200", "name": user_info_data[0].split(': ')[1], "address": user_info_data[1].split(': ')[1], "email": user_info_data[2].split(': ')[1], "birthdate": user_info_data[3].split(': ')[1], "phone_number": user_info_data[4].split(': ')[1], "brain_score": user_info_data[6].split(': ')[1], "image": image_base64}
                ),
                status=200,
                mimetype='application/json'
            )
        else:
            return Response(
                response=json.dumps({"Code": "404", "error": "Image not found"}),
                status=404,
                mimetype='application/json'
            )
    except FileNotFoundError:
        return Response(
            response=json.dumps({"Code": "404", "error": "User not found"}),
            status=404,
            mimetype='application/json'
        )

# 모든 사용자 정보 가져오기 엔드포인트
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
                with open(user_info_file, 'r', encoding='utf-8') as f:
                    user_info = f.readlines()
                user_info_data = [line.strip() for line in user_info]
                image_file = os.path.join(user_images_folder, f"{user_id}.jpg")
                if os.path.exists(image_file):
                    with open(image_file, 'rb') as f:
                        image_data = f.read()
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                    print(user_info_data[6].split(': '))
                    all_users_info[user_id] = {"name": user_info_data[0].split(': ')[1], "address": user_info_data[1].split(': ')[1], "email": user_info_data[2].split(': ')[1], "birthdate": user_info_data[3].split(': ')[1], "phone_number": user_info_data[4].split(': ')[1], "brain_score": user_info_data[6].split(': ')[1], "image": image_base64}
                else:
                    all_users_info[user_id] = {"name": user_info_data[0].split(': ')[1], "address": user_info_data[1].split(': ')[1], "email": user_info_data[2].split(': ')[1], "birthdate": user_info_data[3].split(': ')[1], "phone_number": user_info_data[4].split(': ')[1], "brain_score": user_info_data[6].split(': ')[1], "image": ""}
            except FileNotFoundError:
                all_users_info[user_id] = {"error": "User not found"}
        return Response(
            response=json.dumps(all_users_info, ensure_ascii=False),
            status=200,
            mimetype='application/json'
        )

# aqs.txt 파일 경로
aqs_file_path = './brain/aqs.txt'

# aqs.txt 내용 가져오기 엔드포인트
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



# aqs.txt 파일 경로
audio_file_path =  "./audio/audio.wav"

# aqs.txt 내용 가져오기 엔드포인트
# @app.route('/translate/<province>/<user_id>', methods=['POST'])
# def translate_user(province, user_id):
#     try:   
#         encoded_audio = request.json.get('audio')
#         new_encoded_audio = encoded_audio + '=' * (4 - len(encoded_audio) % 4)
#         audio_data = base64.b64decode(str(new_encoded_audio))
#         print(new_encoded_audio)
#         os.chmod(audio_file_path, 0o777)
#         f = open(audio_file_path, 'wb')
#         f.write(audio_data)
#         f.close()
#         province_code = ""
#         if province == "1":
#             province_code= 'gs'
#         elif province == "2":
#             province_code= 'gw'

#         pretrained_model_src_dir = 'pretrained-model-src'
#         pretrained_model_save_dir = 'pretrained-model-save'

#         source = os.path.join(pretrained_model_src_dir, province_code).replace('\\', '/')
#         savedir = os.path.join(pretrained_model_save_dir, province_code).replace('\\', '/')

#         print("source : ", source)
#         print("savedir : ", savedir)

#         for root, dirs, files in os.walk(source):
#             for file in files:
#                 file_path = os.path.join(root, file)
#                 os.chmod(file_path, 0o777)  # 모든 사용자에게 읽기, 쓰기, 실행 권한을 부여합니다.

#         for root, dirs, files in os.walk(savedir):
#             for file in files:
#                 file_path = os.path.join(root, file)
#                 os.chmod(file_path, 0o777)  # 모든 사용자에게 읽기, 쓰기, 실행 권한을 부여합니다.


#         asr_model = EncoderDecoderASR.from_hparams(
#             source=source,                            
#             savedir=savedir,
#             run_opts={"device":"cpu"}
#         )

#         # Process the audio file using the dummy translate function
#         translated_data = asr_model.transcribe_file(audio_file_path)
        
#         # Brain Score Update
#         script_name = r'C:\Workspace\1_Univ\CAU\3-1\Capstone\Server\BrainDisorder\app.py'
#         command = [r"C:\Python311\python.exe", script_name, audio_file_path]
#         result = subprocess.run(command, capture_output=True, text=True)
#         normal_score, brain_score = extract_floats(result)
#         score = (100-normal_score + brain_score)/2

#         # Store a score of user
#         user_info_file = os.path.join(user_data_folder, f"{user_id}.txt")
#         with open(user_info_file, 'r', encoding='utf-8') as f:
#             user_info = f.readlines()
#         user_info[-1] = "Brain Score: " + str(score) + '\n'
#         with open(file_path, 'w') as file:
#             file.writelines(user_info)
        
#         translate_request = { "text":translated_data }
#         json_request = json.dumps(translate_request)
#         if province_code == 'gs':
#             response = request.post("https://127.0.0.1:7778/translation/to_sd", data=json_request, headers={'Content-Type': 'application/json'})
#             translated_data = response.text
#         else:
#             response = request.post("https://127.0.0.1:7777/translation/to_sd", data=json_request, headers={'Content-Type': 'application/json'})
#             translated_data = response.text
#         translated_data = json.loads(translated_data)
        
#         return jsonify({"Code": 200, "data": translated_data["results"]["translation"]})
 
        
#     except FileNotFoundError as e:
#         return jsonify({"Code": 404, "error": str(e)}), 404
#     except UnicodeDecodeError as e:
#         return jsonify({"Code": 500, "error": f"Encoding error: {str(e)}"}), 500
#     finally:
#         # Delete the file after processing
#         if os.path.exists(audio_file_path):
#             os.remove(audio_file_path)
    

import requests
# aqs.txt 내용 가져오기 엔드포인트
@app.route('/translate/<province>', methods=['POST'])
def translate_anony(province):
    try:
        encoded_audio = request.json.get('audio')
        audio_data = base64.b64decode(str(encoded_audio))
        print(audio_data)
        os.chmod("audio", 0o777)
        f = open(audio_file_path, 'w')
        f.write(str(audio_data))
        f.close()
        print("done")
        province_code = ""
        if province == "1":
            province_code= 'gs'
        elif province == "2":
            province_code= 'gw'

        pretrained_model_src_dir = 'pretrained-model-src'
        pretrained_model_save_dir = 'pretrained-model-save'

        source = os.path.join(pretrained_model_src_dir, province_code).replace('\\', '/')
        savedir = os.path.join(pretrained_model_save_dir, province_code).replace('\\', '/')

        print("source : "+ source)
        print("savedir : "+ savedir)

        for root, dirs, files in os.walk(source):
            for file in files:
                file_path = os.path.join(root, file)
                os.chmod(file_path, 0o777)  # 모든 사용자에게 읽기, 쓰기, 실행 권한을 부여합니다.

        for root, dirs, files in os.walk(savedir):
            for file in files:
                file_path = os.path.join(root, file)
                os.chmod(file_path, 0o777)  # 모든 사용자에게 읽기, 쓰기, 실행 권한을 부여합니다.


        asr_model = EncoderDecoderASR.from_hparams(
            source=source,                            
            savedir=savedir,
            run_opts={"device":"cpu"}
        )
        print("before process asr")
        # Process the audio file using the dummy translate function
        translated_data = asr_model.transcribe_file(audio_file_path)
        translate_request = { "text":translated_data }
        json_request = json.dumps(translate_request)
        if province_code == 'gs':
            response = request.post("http://127.0.0.1:7778/translation/to_sd", data=json_request, headers={'Content-Type': 'application/json'})
            translated_data = response.text
        else:
            response = request.post("http://127.0.0.1:7777/translation/to_sd", data=json_request, headers={'Content-Type': 'application/json'})
            translated_data = response.text
        translated_data = json.loads(translated_data)
        
        # Delete the file after processing
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)
        
        return jsonify({"Code": 200, "data": translated_data["results"]["translation"]})
    except FileNotFoundError as e:
        return jsonify({"Code": 404, "error": str(e)}), 404
    except UnicodeDecodeError as e:
        return jsonify({"Code": 500, "error": f"Encoding error: {str(e)}"}), 500
@app.route('/doctor/<id>', methods=['GET'])
def get_doctor_file(id):
    try:
        # 파일 경로 생성
        file_path = os.path.join('doctor', f'{id}.txt')
        
        # 파일 존재 여부 확인
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # 파일을 줄 단위로 utf-8 인코딩으로 읽기
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # 줄 단위로 내용을 배열로 반환
        return jsonify({'lines': [line.strip() for line in lines]}), 200
    except Exception as e:
        # 에러가 발생한 경우 에러 메시지 반환
        return jsonify({'error': str(e)}), 500
from time import sleep
# wav 파일 업로딩 : 유저 할당 
@app.route('/translate_file/<province>', methods=['POST'])
def translate_file_anony(province):
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        # 파일을 저장하거나 처리
        os.chmod("audio", 0o777)
        file.save(audio_file_path)
        print(audio_file_path)
        print("done")
        province_code = ""
        if province == "1":
            province_code= 'gs'
        elif province == "2":
            province_code= 'gw'

        pretrained_model_src_dir = 'pretrained-model-src'
        pretrained_model_save_dir = 'pretrained-model-save'

        source = os.path.join(pretrained_model_src_dir, province_code).replace('\\', '/')
        savedir = os.path.join(pretrained_model_save_dir, province_code).replace('\\', '/')

        print("source : "+ source)
        print("savedir : "+ savedir)

        for root, dirs, files in os.walk(source):
            for file_temp in files:
                file_path = os.path.join(root, file_temp)
                os.chmod(file_path, 0o777)  # 모든 사용자에게 읽기, 쓰기, 실행 권한을 부여합니다.

        for root, dirs, files in os.walk(savedir):
            for file_temp in files:
                file_path = os.path.join(root, file_temp)
                os.chmod(file_path, 0o777)  # 모든 사용자에게 읽기, 쓰기, 실행 권한을 부여합니다.


        asr_model = EncoderDecoderASR.from_hparams(
            source=source,                            
            savedir=savedir,
            run_opts={"device":"cpu"}
        )
        print("before process asr : " + audio_file_path)
        # Process the audio file using the dummy translate function
        translated_data = asr_model.transcribe_file(audio_file_path)
        print(translated_data)
        translate_request = { "text":translated_data }
        json_request = json.dumps(translate_request)
        print(json_request)
        if province_code == 'gs':
            response = requests.post("http://127.0.0.1:7778/translation/to_sd", data=json_request, headers={'Content-Type': 'application/json'})
            translated_data = response.text
        else:
            response = requests.post("http://127.0.0.1:7777/translation/to_sd", data=json_request, headers={'Content-Type': 'application/json'})
            translated_data = response.text
        translated_data = json.loads(translated_data)
        
        # Delete the file after processing
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)
        
        return jsonify({"Code": 200, "data": translated_data["results"]["translation"]})

# wav 파일 업로딩 : 유저 할당
import threading
def UpdateBrainScore(user_id):
        # Brain Score Update
        script_name = r'C:\Workspace\1_Univ\CAU\3-1\Capstone\Server\BrainDisorder\app.py'
        command = [r"C:\Python311\python.exe", script_name, user_id]
        result = subprocess.run(command)
import shutil
@app.route('/translate_file/<province>/<user_id>', methods=['POST'])
def translate_file(province, user_id):
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        # 파일을 저장하거나 처리
        os.chmod("audio", 0o777)
        file.save(audio_file_path)
        shutil.copyfile(r"C:\Workspace\1_Univ\CAU\3-1\Capstone\Server\audio\audio.wav", r"C:\Workspace\1_Univ\CAU\3-1\Capstone\Server\audio\brain.wav")
        print(type(file))
        province_code = ""
        if province == "1":
            province_code= 'gs'
        elif province == "2":
            province_code= 'gw'

        pretrained_model_src_dir = 'pretrained-model-src'
        pretrained_model_save_dir = 'pretrained-model-save'

        source = os.path.join(pretrained_model_src_dir, province_code).replace('\\', '/')
        savedir = os.path.join(pretrained_model_save_dir, province_code).replace('\\', '/')

        print("source : "+ source)
        print("savedir : "+ savedir)

        for root, dirs, files in os.walk(source):
            for file in files:
                file_path = os.path.join(root, file)
                os.chmod(file_path, 0o777)  # 모든 사용자에게 읽기, 쓰기, 실행 권한을 부여합니다.

        for root, dirs, files in os.walk(savedir):
            for file in files:
                file_path = os.path.join(root, file)
                os.chmod(file_path, 0o777)  # 모든 사용자에게 읽기, 쓰기, 실행 권한을 부여합니다.

        brain_thread = threading.Thread(target=UpdateBrainScore, args=(user_id))
        brain_thread.start()

        asr_model = EncoderDecoderASR.from_hparams(
            source=source,                            
            savedir=savedir,
            run_opts={"device":"cpu"}
        )
        # Process the audio file using the dummy translate function
        translated_data = asr_model.transcribe_file(audio_file_path)
        print(translated_data)
        translate_request = { "text":translated_data }
        json_request = json.dumps(translate_request)
        print(json_request)
        if province_code == 'gs':
            response = requests.post("http://127.0.0.1:7778/translation/to_sd", data=json_request, headers={'Content-Type': 'application/json'})
            translated_data = response.text
        else:
            response = requests.post("http://127.0.0.1:7777/translation/to_sd", data=json_request, headers={'Content-Type': 'application/json'})
            translated_data = response.text
        translated_data = json.loads(translated_data)
        
        
        # Delete the file after processing
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)
        
        return jsonify({"Code": 200, "data": translated_data["results"]["translation"]})

    

#from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing import image
#import numpy as np
#from io import BytesIO
#from PIL import Image


#loaded_model = load_model("./teeth/model.h5")

#@app.route('/predict', methods=['POST'])
#def predict():
#    # 요청에서 base64로 인코딩된 이미지 데이터 받기
#    data = request.json
#    if 'image' not in data:
#        return jsonify({'error': 'No image provided'}), 400
    
#    # base64 디코딩
#    img_data = base64.b64decode(data['image'])
#    img = Image.open(BytesIO(img_data))
#    img = img.resize((64, 64))  # 이미지 크기 조정

#    # 이미지 전처리
#    img_array = image.img_to_array(img)
#    img_array = np.expand_dims(img_array, axis=0)
#    img_array /= 255.0

#    # 모델 예측
#    predictions = loaded_model.predict(img_array)
#    prediction = predictions[0, 0]

#    # 결과 반환
#    result = {
#        'prediction': float(prediction),
#        'result': 'Cavity' if prediction < 0.5 else 'No Cavity'
#    }
#    return jsonify(result)




if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = 5000, debug=True)
