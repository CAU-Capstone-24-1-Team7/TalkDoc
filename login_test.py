import unittest
import requests

class TestFlaskApp(unittest.TestCase):
    BASE_URL = "http://127.0.0.1:5000"

    def test_signup_and_login(self):
        # 회원가입 테스트 데이터
        signup_data_1 = {
            "auth": 1,
            "name": "John Doe",
            "ID": "asd",
            "password": "password123",
            "patient_id": "patient123"
        }

        signup_data_2 = {
            "auth": 2,
            "name": "Jane Smith",
            "ID": "janesmith",
            "password": "password456",
            "patient_id": "patient456"
        }

        # 회원가입 요청 (auth가 1인 경우, 환자 파일이 존재해야 함)
        response = requests.post(f"{self.BASE_URL}/signup", json=signup_data_1)
        self.assertEqual(response.status_code, 404)

        # 환자 파일 생성
        with open('user_data/patient123.txt', 'w') as f:
            f.write("Patient data")

        # 회원가입 요청 (환자 파일이 존재하는 경우)
        response = requests.post(f"{self.BASE_URL}/signup", json=signup_data_1)
        self.assertEqual(response.status_code, 200)

        # 회원가입 요청 (auth가 2인 경우)
        response = requests.post(f"{self.BASE_URL}/signup", json=signup_data_2)
        self.assertEqual(response.status_code, 200)

        # ID 중복 회원가입 요청
        response = requests.post(f"{self.BASE_URL}/signup", json=signup_data_1)
        self.assertEqual(response.status_code, 400)

        # 로그인 테스트 데이터
        login_data = {
            "ID": "asd",
            "password": "password123"
        }

        # 로그인 요청 (성공)
        response = requests.post(f"{self.BASE_URL}/login", json=login_data)
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertEqual(response_data['status_code'], 200)
        self.assertEqual(response_data['auth'], '1')
        self.assertEqual(response_data['name'], 'John Doe')

        # 로그인 요청 (잘못된 비밀번호)
        login_data['password'] = 'wrongpassword'
        response = requests.post(f"{self.BASE_URL}/login", json=login_data)
        self.assertEqual(response.status_code, 400)

        # 로그인 요청 (존재하지 않는 ID)
        login_data['ID'] = 'nonexistent'
        response = requests.post(f"{self.BASE_URL}/login", json=login_data)
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()
