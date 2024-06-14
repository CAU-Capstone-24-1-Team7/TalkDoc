#!/usr/bin/env python3
"""
 * N Park 2022 @ Starcell Inc.
"""

import os
import sys
from speechbrain.pretrained import EncoderDecoderASR

print("start")
provice_code = 'gs'

pretrained_model_src_dir = 'pretrained-model-src'
pretrained_model_save_dir = 'pretrained-model-save'

source = os.path.join(pretrained_model_src_dir, provice_code)
savedir = os.path.join(pretrained_model_save_dir, provice_code)


asr_model = EncoderDecoderASR.from_hparams(
    source=source,                            
    savedir=savedir,
    run_opts={"device":"cuda"}
)

# audio_file = '/data/KsponSpeech/eval_clean_wav/KsponSpeech_E02998.wav'
audio_file = r'C:\Users\CSH\Documents\aa.wav'

#audio_file = sys.argv[1]

print(f'input audio file : {audio_file}')

print(f'ASR output : \n')
print(asr_model.transcribe_file(audio_file))