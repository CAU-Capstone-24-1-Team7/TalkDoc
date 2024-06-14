#!/usr/bin/env python3
"""
 * N Park 2022 @ Starcell Inc.
"""

import os
import sys
from speechbrain.pretrained import EncoderDecoderASR
#from speechbrain.inference.ASR import EncoderDecoderASR
#asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-conformer-transformerlm-ksponspeech", savedir="pretrained_models/asr-conformer-transformerlm-ksponspeech",  run_opts={"device":"cuda"})
#한국어 기본 모델의 예시
provice_code = 'gs'

pretrained_model_src_dir = 'pretrained-model-src'
pretrained_model_save_dir = 'pretrained-model-save'

source = os.path.join(pretrained_model_src_dir, provice_code).replace('\\', '/')
savedir = os.path.join(pretrained_model_save_dir, provice_code).replace('\\', '/')

print("source : ", source)
print("savedir : ", savedir)

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

# audio_file = '/data/KsponSpeech/eval_clean_wav/KsponSpeech_E02998.wav'
audio_file = r'C:\Workspace\1_Univ\CAU\3-1\Capstone\st_set3_collectorgs161_speakergs1066_101_9.wav'

# audio_file = sys.argv[1]

print(f'input audio file : {audio_file}')

print(f'ASR output : \n')
print(asr_model.transcribe_file(audio_file))
