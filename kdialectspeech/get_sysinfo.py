#!/usr/bin/env python

'''
실행법 : python run_pipe.py run_pipe.yaml

speechbrain/utils/superpowers.py 사용 검토
'''

import torch

# print(torch.cuda.is_available())
if torch.cuda.is_available():
    gpu_num = torch.cuda.device_count()
    for gpu in range(gpu_num):
        gpu_model = torch.cuda.get_device_name(gpu)
        mem = torch.cuda.get_device_properties(gpu).total_memory/(1024 * 1024 * 1024)
        print(f'GPU {gpu} - model: {gpu_model}, gpu memory: {mem:.2f} GB')
else:
    print(f'There is no GPU')


