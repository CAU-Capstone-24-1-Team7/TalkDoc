from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# 모델 로드
loaded_model = load_model("model.h5")

# 예측할 이미지 로드 및 전처리
img_path = "teeth_dataset/cavity_1.jpg"

#여기서 대충 전처리 필요
img = image.load_img(img_path, target_size=(64, 64))

img_array = image.img_to_array(img) #배열화
img_array = np.expand_dims(img_array, axis=0) #(1,64,64,3) 64*64 크기의 RGB
img_array /= 255.0 #정규화
#어레이로 변환한거 이제 모델 넘기기

# 모델을 사용하여 예측
predictions = loaded_model.predict(img_array)

# 충치 가능성 출력
print(predictions)

if predictions[0, 0] < 0.5:
    # Cavity인 경우
    print("충치시네요!")
else:
    # No Cavity인 경우
    print("정상이네요!")