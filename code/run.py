import os
import sys
import pathlib

os.environ["DLClight"]="True" # set before importing DLC
import deeplabcut
import numpy as np
import pandas as pd
import tensorflow as tf

import utils

print(f"{os.environ['CUDA_VISIBLE_DEVICES'] = }")

if __name__ == "__main__":

    # process first eye video found
    input_video_file_path: pathlib.Path = next(utils.get_eye_video_paths())
    print(f"Reading video: {input_video_file_path}")

    # phase 1: track points in video and generate h5 file ------------------------- #
    dlc_output_file_path: pathlib.Path = utils.get_dlc_output_path(video_file_path)
    print(f"Writing DLC analysis: {dlc_output_file_path}")
    deeplabcut.analyze_videos(
        config=utils.DLC_PROJECT_PATH / 'config.yaml',
        videos=[
          input_video_file_path.as_posix(),
        ],
        destfolder=dlc_output_file_path.parent.as_posix(),
    )

    # phase 2: fit ellipses to eye perimeter, pupil, and corneal reflection ------- #
    ellipse_output_file_path = dlc_output_file_path.parent / 'ellipse.h5'
    print(f"Writing ellipse fits: {ellipse_output_file_path}")
    utils.fit(
        h5file_path=dlc_output_file_path.as_posix(), 
        ellipse_file_path=ellipse_output_file_path.as_posix(),
    )

    # TODO compute pupil area timeseries
    # TODO store unobserved pupil frames (blinks or stress)
    # TODO output DLC annotation timeseries to NWB.acquisition
    # TODO append BehaviorSeries/Events to NWB
    # TODO use optional rig info to compute gaze location on monitor
    # TODO create asset from results