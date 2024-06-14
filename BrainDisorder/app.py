# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 15:58:19 2022

@author: user
"""

"""
Created on Mon Feb 21 18:29:08 2022

@author: user
"""
import classification
import sys
import numpy as np
import classification
import os
# Connect to Redis
def main():
        user_id = sys.argv[1]
        prediction = classification.model_classification(r'C:\Workspace\1_Univ\CAU\3-1\Capstone\Server\audio\brain.wav')
        score = (100-prediction[-1][0] + prediction[-1][1])/2
        print(score)
        if os.path.exists(r'C:\Workspace\1_Univ\CAU\3-1\Capstone\Server\audio\brain.wav'):
            os.remove(r'C:\Workspace\1_Univ\CAU\3-1\Capstone\Server\audio\brain.wav')
        user_info_file = os.path.join(r'C:\Workspace\1_Univ\CAU\3-1\Capstone\Server\user_data', f"{user_id}.txt")
        print(user_info_file)
        os.chmod(user_info_file, 0o777)
        with open(user_info_file, 'r', encoding='UTF-8') as f:
            user_info = f.readlines()
        user_info[-1] = "Brain Score: " + str(score) + '\n'
        with open(user_info_file, 'w', encoding='UTF-8') as file:
            file.writelines(user_info)
        print("Done")
        exit()
        
if __name__ == "__main__":
    main()