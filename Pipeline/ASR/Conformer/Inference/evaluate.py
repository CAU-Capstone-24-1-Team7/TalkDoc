#!/usr/bin/env python3

import os
import sys
import pandas as pd
import torch
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.utils.metric_stats import ErrorRateStats
import IPython.display as ipd

provice_code = 'kspon'

pretrained_model_src_dir = 'pretrained-model-src'
pretrained_model_save_dir = 'pretrained-model-save'

source = os.path.join(pretrained_model_src_dir, provice_code)
savedir = os.path.join(pretrained_model_save_dir, provice_code)

asr_model = EncoderDecoderASR.from_hparams(
    source=source,
    savedir=savedir,
    run_opts={"device":"cuda"}
)

data_file = '/workspace/speechbrain/recipes/KdialectSpeech/Tokenizer/results/data_prepared/gs/total.csv'
data_df = pd.read_csv(data_file)

num_in_group = 1000
num_group = len(data_df)//num_in_group
# num_group = 10

for gid in range(num_group):
# for gid in range(10):
    wer_filre = 'wer_file_' + str(gid) + '.txt'
    # data_df[gid:gid+10]
    wer_stats = ErrorRateStats()

    for id, row in data_df[gid*num_in_group:(gid+1)*num_in_group].iterrows():
        # print(f'id : {id}')
        # print(f'ID : {row.ID}')
        # print(f'wrd : {row.wrd}')
            id = row.ID
            hyp = [asr_model.transcribe_file(row.wav).split()]
            ref = [row.wrd.split()]

            wer_stats.append(
                ids=[id],
                predict=hyp,
                target=ref
            )
            # with open(wer_filre, "a") as w:
            #     wer_stats.write_stats(w) 
    with open(wer_filre, "a") as w:
        wer_stats.write_stats(w)


      
    
