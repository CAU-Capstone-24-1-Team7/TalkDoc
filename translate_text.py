import unittest
from unittest.mock import patch, mock_open, MagicMock
from flask import Flask, jsonify
import base64
import os
import subprocess

# Assuming the Flask app is in a file named 'app.py'
from app import app

class TranslateUserTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def encode_wav_file(self, file_path):
        with open(file_path, 'rb') as wav_file:
            wav_data = wav_file.read()
        encoded_audio = base64.b64encode(wav_data).decode('utf-8')
        return encoded_audio

    @patch('app.os.path.exists', return_value=True)
    @patch('app.os.remove')
    @patch('app.os.chmod')
    @patch('app.os.walk')
    @patch('app.open', new_callable=mock_open, read_data='User Data\nBrain Score: 0\n')
    @patch('app.subprocess.run')
    def test_translate_user_success(self, mock_subprocess_run, mock_open, mock_os_walk, mock_os_chmod, mock_os_remove, mock_os_path_exists):
        # Path to the WAV file
        wav_file_path = 'path/to/your/test.wav'
        encoded_audio = self.encode_wav_file(wav_file_path)

        mock_subprocess_run.return_value = MagicMock(stdout='Normal: 50\nBrain: 50')
        mock_os_walk.side_effect = [
            [('/fake/src', [], ['file1', 'file2'])],
            [('/fake/dst', [], ['file3', 'file4'])],
        ]

        response = self.app.get(f'/translate/1/123?audio={encoded_audio}')

        assert response.status_code == 200
        assert 'data' in response.json

    @patch('app.os.path.exists', return_value=False)
    def test_translate_user_file_not_found(self, mock_os_path_exists):
        # Path to the WAV file
        wav_file_path = r'C:\Workspace\1_Univ\CAU\3-1\Capstone\st_set3_collectorgs161_speakergs1066_101_9.wav'
        encoded_audio = self.encode_wav_file(wav_file_path)

        response = self.app.get(f'/translate/1/123?audio={encoded_audio}')

        assert response.status_code == 404
        assert response.json['error'] == 'File not found'

    @patch('app.os.path.exists', return_value=True)
    @patch('app.base64.b64decode', side_effect=UnicodeDecodeError('codec', b'\x00\x00', 1, 2, 'reason'))
    def test_translate_user_unicode_decode_error(self, mock_b64decode, mock_os_path_exists):
        # Path to the WAV file
        wav_file_path = 'path/to/your/test.wav'
        encoded_audio = self.encode_wav_file(wav_file_path)

        response = self.app.get(f'/translate/1/123?audio={encoded_audio}')

        assert response.status_code == 500
        assert 'Encoding error' in response.json['error']

if __name__ == '__main__':
    unittest.main()
