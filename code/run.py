import os
import sys
import pathlib

os.environ["DLClight"]="True" # set before importing DLC
import deeplabcut
import numpy as np
import pandas as pd
import tensorflow as tf

import utils

if __name__ == "__main__":

    # process first eye video found
    input_video_file_path: pathlib.Path = next(
        utils.get_eye_video_paths()
        )
    print(f"Reading video: {input_video_file_path}")
    """
    # phase 1: track points in video and generate h5 file ------------------------- #
    print(f"Writing DLC analysis: {utils.RESULTS_PATH}")
    
    deeplabcut.analyze_videos(
        config=utils.DLC_PROJECT_PATH / 'config.yaml',
        videos=[
          input_video_file_path.as_posix(),
        ],
        destfolder=utils.RESULTS_PATH.as_posix(),
    )
    """

    # phase 2: fit ellipses to eye perimeter, pupil, and corneal reflection ------- #
    ellipse_output_file_path = utils.RESULTS_PATH / 'ellipse.h5'
    print(f"Writing ellipse fits: {ellipse_output_file_path}")
    utils.process_ellipses(
        h5file_path=(
            utils.get_dlc_output_path(
                input_video_file_path=input_video_file_path,
                dlc_destfolder=utils.DATA_PATH,
                )
            .as_posix()
        ),
        ellipse_file_path=ellipse_output_file_path.as_posix(),
    )
