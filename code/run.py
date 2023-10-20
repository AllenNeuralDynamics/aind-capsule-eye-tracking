import os
import sys
import pathlib

os.environ["DLClight"]="True"

import deeplabcut
import numpy as np
import pandas as pd
import tensorflow as tf

import utils

cudadev=os.system("cat $PBS_GPUFILE | rev | cut -d\"u\" -f1")
os.environ['CUDA_VISIBLE_DEVICES'] = str(cudadev)
print(f"{os.environ['CUDA_VISIBLE_DEVICES'] = "})

if __name__ == "__main__":
    # phase 1 
    video_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    output_file = video_file_path[:-4] + 'DeepCut_resnet50_universal_eye_trackingJul10shuffle1_1030000.h5'

    # ### Path to trained model:
    path_config_file = 'code/config.yaml'

    # ### Track points in video and generate h5 file:
    deeplabcut.analyze_videos(path_config_file,[video_file_path]) #can take a list of input videos

    os.rename(output_file,output_file_path)

    # phase 2
    h5file_path = output_file_path
    ellipse_file_path = f'results/ellipse.h5'
    utils.fit(h5file_path) 
