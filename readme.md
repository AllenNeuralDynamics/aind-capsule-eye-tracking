# aind-capsule-eye-tracking
*under development*

## usage
- requires trained DLC model in data asset:
  - **`universal_eye_tracking-peterl-2019-07-10`**
  - `05529cfc-23fe-4ead-9490-71763e9f7c01` 
  - https://codeocean.allenneuraldynamics.org/data-assets/05529cfc-23fe-4ead-9490-71763e9f7c01/universal_eye_tracking-peterl-2019-07-10

- currently runs on first video file found in **`data/`** with **`eye`** in the filename (recursive glob, case-insensitive)

- small 90 second sample in data asset available for testing: 
  - **`eye-tracking-test-video`** 
  - `6a8e6813-f883-4278-b42d-f2e174d760e3`
  - https://codeocean.allenneuraldynamics.org/data-assets/6a8e6813-f883-4278-b42d-f2e174d760e3/behavior-videos

- full length session data and pre-run DLC output h5 data assets:
  - raw data `ecephys_681532_2023-10-18_13-01-15`: 
    - `7bc771ec-c3e6-44b3-b365-78b7e06137cb`
    - https://codeocean.allenneuraldynamics.org/data-assets/7bc771ec-c3e6-44b3-b365-78b7e06137cb/ecephys_681532_2023-10-18_13-01-15
  - DLC output from this capsule (for testing ellipse-fitting alone) `ecephys_681532_2023-10-18_eye-tracking-output`: 
    - `2fd041ed-bd53-4804-8934-36b00356fa8c`
    - https://codeocean.allenneuraldynamics.org/data-assets/2fd041ed-bd53-4804-8934-36b00356fa8c/results
    - enable `REUSE_DLC_OUTPUT_H5_IN_ASSET` in `code/run.py`

## outputs
  - `results/*DLC_resnet50_universal_eye_trackingJul10shuffle1_1030000.h5` file with `(x, y, likelihood)` for each `(cr*, pupil*, eye*)` point in each frame 
  - `results/ellipses.h5` with `(center_x, center_y, width, height, phi)` for ellipses fit to each set of points `(cr*, pupil*, eye*)` 
      - `width`: semi-major axis (`a` in diagram)
      - `height`: semi-minor axis (`b` in diagram)
      - `phi`: counterclockwise rotation of major-axis in radians from x-axis of video
    <img src="https://github.com/AllenNeuralDynamics/aind-capsule-eye-tracking/assets/63425812/76ea30b3-ce45-4c3c-bbac-9dbbfb85ccb3" width="200"/>
  - `results/qc/*.png` various images for checking results

## links
- DLC conda environment: https://github.com/DeepLabCut/DeepLabCut/blob/main/conda-environments/DEEPLABCUT.yaml
- source code for **`deeplabcut.analyze_videos()`**: https://github.com/DeepLabCut/DeepLabCut/blob/main/deeplabcut/pose_estimation_tensorflow/predict_videos.py#L268

## TODO
- fix index error (0 out of bounds, axis 1)
- fix startup warnings
- read rig.json for parameters
  - check for pixel^2 -> cm^2 conversion
- ? output DLC annotation timeseries to NWB.acquisition
- output processed BehaviorSeries/Events to NWB.processing
- store unobserved pupil frames (blinks or stress) in nwb
- ? use rig info to compute gaze location on monitor
- create asset from results automatically

## done
- setup DLC env
- get files and config from lims process
- add DLC project to data asset
- get DLC running
- get ellipse-fitting running
- parallelize ellipse-fitting to use all 16 cores
- output random selection of video frames with annotated points + ellipses overlaid
- compute pupil area timeseries

