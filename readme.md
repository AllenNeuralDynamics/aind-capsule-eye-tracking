# aind-capsule-eye-tracking
*under development*

- uses trained DLC model in data asset:
  **`universal_eye_tracking-peterl-2019-07-10`**
  - `05529cfc-23fe-4ead-9490-71763e9f7c01` 
  - https://codeocean.allenneuraldynamics.org/data-assets/05529cfc-23fe-4ead-9490-71763e9f7c01/universal_eye_tracking-peterl-2019-07-10

- currently runs on first video file found in **`data/`** with **`eye`** in the filename (recursive glob, case-insensitive)
- outputs:
  - `results/*DLC_resnet50_universal_eye_trackingJul10shuffle1_1030000.h5` file with `(x, y, likelihood)` for each `(cr*, pupil*, eye*)` point in each frame 
  - `results/ellipses.h5` with `(center_x, center_y, width, height, phi)` for
    ellipses fit to each set of points `(cr*, pupil*, eye*)` 
      - `width`: major axis
      - `height`: minor axis
      - `phi`: counterclockwise rotation of major-axis in radians from x-axis
  - `results/qc/*.png` evenly-spaced video frames throughout 
  
## links
- DLC conda environment: https://github.com/DeepLabCut/DeepLabCut/blob/main/conda-environments/DEEPLABCUT.yaml
- source code for **`deeplabcut.analyze_videos()`**: https://github.com/DeepLabCut/DeepLabCut/blob/main/deeplabcut/pose_estimation_tensorflow/predict_videos.py#L268

## TODO
- fix startup warnings
- output random selection of video frames with annotated points + ellipses overlaid
- compute pupil area timeseries
- store unobserved pupil frames (blinks or stress)
- ? output DLC annotation timeseries to NWB.acquisition
- output processed BehaviorSeries/Events to NWB.processing
- ? use rig info to compute gaze location on monitor
- create asset from results
