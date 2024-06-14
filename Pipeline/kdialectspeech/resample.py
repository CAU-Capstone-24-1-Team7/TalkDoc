#!/usr/bin/env python3
"""오디오 파일을 리샘플링하는 프로그램, 
확인 작업 필요
Author
 * N Park 2022
"""

from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
import librosa
import soundfile as sf
import logging

def read_audiosample(audio_dict_file):
    with open(audio_dict_file, 'r') as f:
        lines = f.readlines()
        file_sr_dict = {}
        for l in lines:
            file_sr = l.split(':')
            # print(file_sr)
            file_sr_dict[file_sr[0]] = int(file_sr[1].strip())
        f.close()
        
    # print(file_sr_dict)
    return file_sr_dict

def resample_audio(audio_dict_file='wrong_samplerate.txt', smaplerate=16000):    
    # logger.info(f'{audio_dict_file}의 파일들을 변환 시작-----')
    print(f'{audio_dict_file}의 파일들을 변환 시작-----')
    for file, sr in read_audiosample(audio_dict_file).items():
        y, sr = librosa.load(file, sr=sr)
        resample = librosa.resample(y, sr, smaplerate)
        sf.write(file, resample, smaplerate, format='WAV', endian='LITTLE', subtype='PCM_16')

                
if __name__ == '__main__':

    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    ##### setup logging
    logger = logging.getLogger(__name__)

    log_config = hparams["log_config"]
    log_file = 'resample_' + hparams["log_file"]

    logger_overrides = {
        "handlers": {"file_handler": {"filename": log_file}}
    }
    #####

    wrong_samplerate_file = hparams["wrong_samplerate_file"]
    logger.info(f'{wrong_samplerate_file}의 파일들을 변환 시작-----')

    resample_audio(wrong_samplerate_file, smaplerate=16000)

    logger.info(f'{wrong_samplerate_file}의 파일들을 변환 종료-----')
