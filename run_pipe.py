#!/usr/bin/env python

'''
실행법 : python run_ml.py run_ml.yaml

speechbrain/utils/superpowers.py 사용 검토
'''

import logging
import sys
import os
import subprocess
import glob
import shutil

from pathlib import Path

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

from kdialectspeech.resample import resample_audio


if __name__ == "__main__":
    # run-pipe.yaml 파일의 경로
    yaml_file_path = os.path.join(os.path.dirname(__file__), "run_pipe.yaml")
    
    # CLI:
    # hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    hparams_file, run_opts, overrides = sb.parse_arguments([yaml_file_path])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)


    ##### setup logging
    logger = logging.getLogger(__name__)
    log_config = hparams["log_config"]
    log_file = hparams["log_file"]
    logger_overrides = {
        "handlers": {"file_handler": {"filename": log_file}}
    }
    sb.utils.logger.setup_logging(log_config, logger_overrides)
    #####

    ##### 데이터 경로 설정
    data_path = 'C:/Users/CSH/Desktop/cap/data/allset'
    hparams["data_save_path"] = data_path  # 데이터 경로를 설정합니다.
    logger.info(f'Data path set to: {data_path}')
    #####
    
    ##### 방언별 실행 : 토크나이저, 언어모델, 음성인식모델
    run_provinces = hparams["run_provinces"]
    gpu_num = hparams["gpu_num"]
    pretrained_model_base = hparams['pretrained_model_base']

    for run_province in run_provinces:
        province_option = '--province_code=' + run_province
        pretrained_model_dir = os.path.join(pretrained_model_base, run_province)
        logger.info(f'make pretrained_model_dir : {pretrained_model_dir}')
        os.makedirs(pretrained_model_dir, exist_ok=True)

        ##### 토크나이저 실행(방언별로 각각 실행)
        if 'tokenizer' in hparams['run_modules']:
            # 토크나이저 실행 코드
            logger.info(f'tokenizer run_option : {province_option}')
            tokenizer_dir = hparams["tokenizer_dir"]
            logger.info(f'tokenizer_dir : {tokenizer_dir}')

            if run_province == 'jj':
                cmd = 'python train.py hparams/1K_unigram_subword_bpe_jj.yaml ' + province_option
            else:
                cmd = 'python train.py hparams/5K_unigram_subword_bpe.yaml ' + province_option
            
            logger.info(f'cmd : {cmd}')

            pwd = os.getcwd()
             
            result = subprocess.run(cmd,
                text=True,
                cwd=tokenizer_dir,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            logger.info(f'returncode : {result.returncode}')
            if result.returncode == 0:
                logger.info(f'tokenizer completed successfully')
                if hparams['copy_trained_model']:
                    token_result_dir = os.path.join(tokenizer_dir, 'results/data_prepared/' + run_province)
                    if run_province == 'jj': 
                        tokenizer =  os.path.join(token_result_dir, '1000_unigram.model')
                    else:
                        tokenizer =  os.path.join(token_result_dir, '5000_unigram.model')
                    tokenizer_target = os.path.join(pretrained_model_dir, 'tokenizer.ckpt')
                    logger.info(f'tokenizer file copy from {tokenizer} to {tokenizer_target}')
                    shutil.copy(tokenizer, tokenizer_target)
            
        #####

        ##### 언어모델 실행(방언별로 각각 실행)
        if 'lm' in hparams['run_modules']:
            # 언어모델 실행 코드
            logger.info(f'lm run_option : {province_option}')
            logger.info(f'gpu num : {gpu_num}')
            lm_dir = hparams["lm_dir"]
            if gpu_num > 1:
                cmd = "python -m torch.distributed.launch --nproc_per_node=" \
                    + str(gpu_num) \
                    + " train.py hparams/transformer.yaml --distributed_launch --distributed_backend='nccl' " \
                    + province_option
            else:
                cmd = 'python train.py hparams/transformer.yaml ' + province_option

            logger.info(f'lm_dir : {lm_dir}')

            result = subprocess.run(
                cmd,
                text=True,
                cwd=lm_dir,
                shell=True,
                stdout=subprocess.PIPE
            )

            if result.returncode == 0:
                logger.info(f'LM completed successfully')
                if hparams['copy_trained_model']:
                    lm_result_dir = os.path.join(lm_dir, 'results/Transformer/5555' + run_province + '/save')
                    lm_model_dir = sorted(glob.glob(lm_result_dir + '/CKPT*'), key=os.path.getmtime)[0]
                    lm_model_dir = Path(lm_model_dir).stem
                    lm_model = os.path.join(lm_result_dir, lm_model_dir + 'model.ckpt')
                    lm_target = os.path.join(pretrained_model_dir, 'lm.ckpt')
                    logger.info(f'lm file copy from {lm_model} to {lm_target}')
                    shutil.copy(lm_model, lm_target)

        #####

        ##### 음성인식 실행(방언별로 각각 실행)
        if 'asr' in hparams['run_modules']:
            # 음성인식 실행 코드
            logger.info(f'asr run_option : {province_option}')
            logger.info(f'gpu num : {gpu_num}')
            logger.info(f'province : {run_province}')
            asr_dir = hparams["asr_dir"]
            if gpu_num > 1:
                cmd = "python -m torch.distributed.launch --nproc_per_node=" \
                    + str(gpu_num) \
                    + " train.py hparams/conformer_medium.yaml --distributed_launch --distributed_backend='nccl' " \
                    + province_option
            else:
                cmd = 'python train.py hparams/conformer_medium.yaml ' + province_option
            logger.info(f'cmd : {cmd}')
            logger.info(f'asr_dir : {asr_dir}')

            result = subprocess.run(
                cmd,
                text=True,
                cwd=asr_dir,
                shell=True,
                stdout=subprocess.PIPE
            )

            if result.returncode == 0:
                logger.info(f'asr completed successfully')
                if hparams['copy_trained_model']:
                    hparam_file = os.path.join(pretrained_model_base, 'hyperparams.yaml')
                    hparam_file_target = os.path.join(pretrained_model_dir, 'hyperparams.yaml')
                    logger.info(f'hparma file copy from {hparam_file} to {hparam_file_target}')
                    shutil.copy(hparam_file, hparam_file_target)

                    train_result_dir = os.path.join(asr_dir, 'results/Conformer/5555/' + run_province + '/save')
                    best_model_dir = sorted(glob.glob(train_result_dir + '/CKPT*'), key=os.path.getmtime)[0]
                    best_model_dir = Path(best_model_dir).stem
                    best_model = os.path.join(train_result_dir, best_model_dir, 'model.ckpt')
                    best_model_target = os.path.join(pretrained_model_dir, 'asr.ckpt')
                    logger.info(f'asr model file copy from {best_model} to {best_model_target}')
                    shutil.copy(best_model, best_model_target)

                    normalizer = os.path.join(train_result_dir, best_model_dir + '/normalizer.ckpt')
                    normalizer_target = os.path.join(pretrained_model_dir, 'normalizer.ckpt')
                    logger.info(f'normalizer file copy from {best_model} to {best_model_target}')
                    shutil.copy(normalizer, normalizer_target)

        #####

def resample_audio_files(audio_file_path, target_sample_rate=16000):
    """
    Resample audio files to the target sample rate using pydub.

    Parameters:
    - audio_file_path (str): The path to the audio file or directory containing audio files.
    - target_sample_rate (int): The target sample rate.

    Returns:
    - None
    """
    # Check if the input is a file or a directory
    if os.path.isfile(audio_file_path):
        audio_files = [audio_file_path]
    elif os.path.isdir(audio_file_path):
        # Get all audio files in the directory
        audio_files = [os.path.join(audio_file_path, f) for f in os.listdir(audio_file_path) if f.endswith('.wav')]
    else:
        raise ValueError("Invalid input: 'audio_file_path' must be a file path or a directory path.")

    # Resample each audio file
    for file_path in audio_files:
        audio = AudioSegment.from_wav(file_path)
        audio = audio.set_frame_rate(target_sample_rate)
        audio.export(file_path, format="wav")