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
DATA = pathlib.Path('/root/capsule/data/')
DLC_PROJECT_PATH = DATA / 'universal_eye_tracking-peterl-2019-07-10'
VIDEO_SUFFIXES = ('.mp4', '.avi', '.wmv', '.mov')

def get_eye_paths() -> Iterator[pathlib.Path]:
    for path in DATA.iterdir():
        if path.name == DLC_PROJECT_PATH.name:
            continue
        if not path.is_dir():
            continue
        yield from (p for p in path.glob('*behavior*/*[eE]ye*') if p.suffix() in VIDEO_SUFFIXES)
            
if __name__ == "__main__":
    # phase 1 
    video_file_path = next(get_eye_paths())
    print(video_file_path.as_posix())

    # output_file = pathlib.Path('results/') / video_file_path.with_suffix(
    #     'DeepCut_resnet50_universal_eye_trackingJul10shuffle1_1030000.h5'
    #     ).name

    # # ### Path to trained model:
    # # ### Track points in video and generate h5 file:
    deeplabcut.analyze_videos(
        config=DLC_PROJECT_PATH / 'config.yaml',
        videos=[
          video_file_path.as_posix(),
        ],
    )

    # os.rename(output_file,output_file_path)

    # # phase 2
    # h5file_path = output_file_path
    # ellipse_file_path = f'results/ellipse.h5'
    # utils.fit(h5file_path) 
