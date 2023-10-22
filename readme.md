# aind-capsule-eye-tracking
*under development*

- uses trained DLC model in data asset:
  - **`universal_eye_tracking-peterl-2019-07-10`**
  - `05529cfc-23fe-4ead-9490-71763e9f7c01` 
  - https://codeocean.allenneuraldynamics.org/data-assets/05529cfc-23fe-4ead-9490-71763e9f7c01/universal_eye_tracking-peterl-2019-07-10

- currently runs on first video file found in **`data/`** with **`eye`** in the filename (recursive glob, case-insensitive)

- small 90 second sample in data asset available for testing: 
  - **`eye-tracking-test-video`** 
  - `6a8e6813-f883-4278-b42d-f2e174d760e3`
  - https://codeocean.allenneuraldynamics.org/data-assets/6a8e6813-f883-4278-b42d-f2e174d760e3/behavior-videos
  
## links
- DLC conda environment: https://github.com/DeepLabCut/DeepLabCut/blob/main/conda-environments/DEEPLABCUT.yaml
- source code for **`deeplabcut.analyze_videos()`**: https://github.com/DeepLabCut/DeepLabCut/blob/main/deeplabcut/pose_estimation_tensorflow/predict_videos.py#L268

## TODO
- fix startup warnings
- read rig.json for parameters
  - check for pixel^2 -> cm^2 conversion
- ? output DLC annotation timeseries to NWB.acquisition
- output processed BehaviorSeries/Events to NWB.processing
- store unobserved pupil frames (blinks or stress) in nwb
- ? use rig info to compute gaze location on monitor
- create asset from results

## done
- output random selection of video frames with annotated points + ellipses overlaid
- compute pupil area timeseries
