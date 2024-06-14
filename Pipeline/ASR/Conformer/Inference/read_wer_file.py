#!/usr/bin/env python

import glob

path = './wer*.txt'

file_list = glob.glob(path)

for file in file_list:
    f = open(file)
    print(file)
    print(f.readline())
