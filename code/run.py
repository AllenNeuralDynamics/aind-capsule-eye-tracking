import os
import sys
import pathlib

os.environ["DLClight"]="True" # set before importing DLC
import deeplabcut
import numpy as np
import pandas as pd
import tensorflow as tf

import utils
import qc

if __name__ == "__main__":

    # process first eye video found
    input_video_file_path: pathlib.Path = next(
        utils.get_eye_video_paths()
        )
    print(f"Reading video: {input_video_file_path}")
    
    # phase 1: track points in video and generate h5 file ------------------------- #
    print(f"Writing DLC analysis: {utils.RESULTS_PATH}")
    
    deeplabcut.analyze_videos(
        config=utils.DLC_PROJECT_PATH / 'config.yaml',
        videos=[
          input_video_file_path.as_posix(),
        ],
        destfolder=utils.RESULTS_PATH.as_posix(),
    )
    dlc_output_h5_path = utils.get_dlc_output_h5_path(
                input_video_file_path=input_video_file_path,
                output_dir_path=utils.RESULTS_PATH,
        )

    # phase 2: fit ellipses to eye perimeter, pupil, and corneal reflection ------- #
    output_file_path = utils.RESULTS_PATH / 'ellipses.h5'
    print(f"Writing ellipse fits: {output_file_path}")
    body_part_to_df = utils.process_ellipses(
        dlc_output_h5_path=dlc_output_h5_path,
        output_file_path=output_file_path,
    )
    
    # qc plots -------------------------------------------------------------------- #
    # plot ellipses on example frames 
    NUM_FRAMES = 5
    QC_PATH = utils.RESULTS_PATH / "qc" 
    QC_PATH.mkdir(exist_ok=True, parents=True)
    print(f"Writing {NUM_FRAMES} example frames to {QC_PATH}")
    total_frames = utils.get_video_frame_count(input_video_file_path)
    step = total_frames // NUM_FRAMES + 1
    for idx in range(step//2, total_frames, step): # avoid frames at the very start/end
        fig = qc.plot_video_frame_with_ellipses(
            video_path=input_video_file_path,
            all_ellipses=body_part_to_df,
            frame_index=idx,
            dlc_output_h5_path=dlc_output_h5_path,
        )
        fig.savefig(
            QC_PATH / f"{input_video_file_path.stem}_{idx}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
    
    # plot path of fitted pupil center
    qc.plot_video_frame_with_pupil_path(
        video_path=input_video_file_path,
        pupil_ellipses=body_part_to_df['pupil'],
        ).savefig(
            QC_PATH / f"{input_video_file_path.stem}_pupil_path.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )