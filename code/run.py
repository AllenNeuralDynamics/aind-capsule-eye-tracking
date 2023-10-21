import os
import sys
import pathlib
from typing import Iterator

os.environ["DLClight"]="True"

import deeplabcut
import numpy as np
import pandas as pd
import tensorflow as tf

import utils

# cudadev=os.system("cat $PBS_GPUFILE | rev | cut -d\"u\" -f1")
# os.environ['CUDA_VISIBLE_DEVICES'] = str(cudadev)
# print(cudadev)

DLC_PROJECT_PATH = '/data/universal_eye_tracking-peterl-2019-07-10'

def get_eye_paths() -> Iterator[pathlib.Path]:
    for asset in pathlib.Path('/data').iterdir():
        if asset.name == pathlib.Path(DLC_PROJECT_PATH).name:
            continue
        if not asset.is_dir():
            continue
        yield from asset.glob('*behavior*/*[eE]ye*')
            
if __name__ == "__main__":
    # phase 1 
    video_file_path = next(file for file in get_eye_paths() if file.suffix != '.json')
    print(video_file_path)

    # output_file = pathlib.Path('results/') / video_file_path.with_suffix(
    #     'DeepCut_resnet50_universal_eye_trackingJul10shuffle1_1030000.h5'
    #     ).name

    # # ### Path to trained model:
    path_config_file = 'code/config.yaml'

    # # ### Track points in video and generate h5 file:
    deeplabcut.analyze_videos(path_config_file, [video_file_path.as_posix()]) #can take a list of input videos

    # os.rename(output_file,output_file_path)

    # # phase 2
    # h5file_path = output_file_path
    # ellipse_file_path = f'results/ellipse.h5'
    # utils.fit(h5file_path) 
