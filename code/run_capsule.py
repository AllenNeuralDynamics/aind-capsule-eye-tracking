import contextlib
import json
import os
import pathlib
import random

os.environ["DLClight"]="True" # set before importing DLC
import deeplabcut
import numpy as np
import pandas as pd
import qc
import tensorflow as tf
import utils

SAVE_ANNOTATED_VIDEO = False
"""Currently very slow (~3 fps)"""
REUSE_DLC_OUTPUT_H5_IN_ASSET = True
"""Instead of re-generating DLC h5 file, use one in a data asset - for quickly testing
ellipse fitting, qc"""

def main():
    print(tuple(pathlib.Path('/data').glob('*')))
    # process first eye video found
    input_video_file_path: pathlib.Path = next(
        utils.get_video_paths(), None
        )
    

    if input_video_file_path is None:
        print(tuple(utils.DATA_PATH.rglob('*')))
        raise FileNotFoundError(f"No video files found matching {utils.VIDEO_FILE_GLOB_PATTERN=}, {utils.VIDEO_SUFFIXES=}")
    print(f"Reading video: {input_video_file_path}")
    utils.write_json_with_session_id()
    # phase 1: track points in video and generate h5 file ------------------------- #

    if REUSE_DLC_OUTPUT_H5_IN_ASSET:
        # get existing h5 file from data/ 
        temp_files = set()
        with contextlib.suppress(FileNotFoundError):
            existing_h5 = utils.get_dlc_output_h5_path(
                input_video_file_path=input_video_file_path,
                output_dir_path=utils.DATA_PATH,
            ) 
            print(f"{REUSE_DLC_OUTPUT_H5_IN_ASSET=}: using {existing_h5}")
            # - a pickle file exists too
            # - copy everything with matching filename component to results/
            for file in existing_h5.parent.glob(f"{existing_h5.stem}*"):
                temp_files.add(dest := utils.RESULTS_PATH / file.name)
                if not dest.exists(): # during testing we may have already made this 
                    dest.symlink_to(file)
        # no need to skip DLC - it will see the existing h5 and skip itself
 
    print(f"Running DLC analysis and writing to: {utils.RESULTS_PATH}")
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
    print(f"Running ellipse fitting and writing to: {output_file_path}")
    body_part_to_df = utils.run_ellipse_processing(
        dlc_output_h5_path=dlc_output_h5_path,
        output_file_path=output_file_path,
    )
    
    # qc plots -------------------------------------------------------------------- #

    utils.QC_PATH.mkdir(exist_ok=True, parents=True)

    # example frames with ellipses drawn 
    NUM_FRAMES = 5
    print(f"Writing {NUM_FRAMES} example frames to {utils.QC_PATH}")
    total_frames = utils.get_video_frame_count(input_video_file_path)
    step = total_frames // NUM_FRAMES + 1
    for idx in range(step//2, total_frames, step): # avoid frames at the very start/end
        qc.plot_video_frame_with_ellipses(
            video_path=input_video_file_path,
            all_ellipses=body_part_to_df,
            frame_index=idx,
            dlc_output_h5_path=dlc_output_h5_path,
        ).savefig(
            utils.QC_PATH / f"{input_video_file_path.stem}_{idx}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
    
    # path of fitted pupil on a frame
    print(f"Writing example frame with path of pupil center to {utils.QC_PATH}")
    qc.plot_video_frame_with_pupil_path(
        video_path=input_video_file_path,
        pupil_ellipses=body_part_to_df['pupil'],
        ).savefig(
            utils.QC_PATH / f"{input_video_file_path.stem}_pupil_path.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )

    # pupil area timeseries
    print(f"Writing plot of pupil area to {utils.QC_PATH}")
    qc.plot_pupil_area(
        pupil_ellipses=body_part_to_df['pupil'],
        pixel_to_cm=None,
        fps=utils.get_dlc_pickle_metadata(dlc_output_h5_path)['fps'],
        ).savefig(
            utils.QC_PATH / f"{input_video_file_path.stem}_pupil_area.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )

    # frames that didn't meet criteria for fitting
    NUM_FRAMES_PER_ELLIPSE = 5
    print(f"Writing sets of up to {NUM_FRAMES_PER_ELLIPSE} frames that didn't meet criteria for fitting each ellipse type")
    total_frames = utils.get_video_frame_count(input_video_file_path)
    folder = utils.QC_PATH / "failed_ellipse_fits"
    for body_part, df in body_part_to_df.items():
        frames_without_ellipses = np.where(pd.isna(df.center_x))[0]
        if (num_frames := len(frames_without_ellipses)) == 0:
            continue
        json_path = folder / f"{body_part}.json"
        print(f"\t- failed to fit {body_part} ellipses for {num_frames} frames")
        folder.mkdir(exist_ok=True, parents=True)
        print(f"\t- writing frame numbers to {json_path}")
        json_path.write_text(
            json.dumps(
                dict(frames_without_ellipses=frames_without_ellipses.tolist()),
                indent=4,
                )
            )
        random.shuffle(frames_without_ellipses)
        for idx in range(min(num_frames, NUM_FRAMES_PER_ELLIPSE)):
            fig = qc.plot_video_frame_with_dlc_points(
                video_path=input_video_file_path,
                dlc_output_h5_path=dlc_output_h5_path,
                frame_index=frames_without_ellipses[idx],
            )
            fig.suptitle(f"did not meet criteria for ellipse-fitting of {body_part}")
            fig.savefig(
                folder / f"{body_part}_{frames_without_ellipses[idx]}.png",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
            )
    
    # save parameters used
    print("Writing (incomplete) processing.json")
    (utils.RESULTS_PATH / "processing.json").write_text(
        json.dumps({
            "data_processes": {
                "parameters": {
                    "min_liklelihood_threshold": utils.MIN_LIKELIHOOD_THRESHOLD,
                    "min_num_points_for_ellipse_fitting": utils.MIN_NUM_POINTS_FOR_ELLIPSE_FITTING,
                    "num_nan_frames_either_side_of_invalid_eye_frame": utils.NUM_NAN_FRAMES_EITHER_SIDE_OF_INVALID_EYE_FRAME,
                }
            }
        }, indent=4)
    )

    if SAVE_ANNOTATED_VIDEO:
        output_video_path = utils.RESULTS_PATH / input_video_file_path.name
        print(f"Writing annotated video file to {output_video_path}")
        qc.write_video_with_ellipses_and_dlc_points(
            video_path=input_video_file_path, 
            all_ellipses=body_part_to_df,
            dlc_output_h5_path=dlc_output_h5_path,
            dest_path=output_video_path,
        )

    if REUSE_DLC_OUTPUT_H5_IN_ASSET:
        for file in temp_files:
            file.unlink()

    
if __name__ == "__main__":
    main()
