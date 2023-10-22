DLC conda environment: https://github.com/DeepLabCut/DeepLabCut/blob/main/conda-environments/DEEPLABCUT.yaml

source code for **`deeplabcut.analyze_videos()`**: https://github.com/DeepLabCut/DeepLabCut/blob/main/deeplabcut/pose_estimation_tensorflow/predict_videos.py#L268

data asset **`universal_eye_tracking-peterl-2019-07-10`**: `05529cfc-23fe-4ead-9490-71763e9f7c01` 
https://codeocean.allenneuraldynamics.org/data-assets/05529cfc-23fe-4ead-9490-71763e9f7c01/universal_eye_tracking-peterl-2019-07-10

- runs on first video file found in **`data/`** with **`eye`** in the filename (recursive glob, case-insensitive)

## TODO
- compute pupil area timeseries
- store unobserved pupil frames (blinks or stress)
- ? output DLC annotation timeseries to NWB.acquisition
- output processed BehaviorSeries/Events to NWB.processing
- ? use rig info to compute gaze location on monitor
- create asset from results