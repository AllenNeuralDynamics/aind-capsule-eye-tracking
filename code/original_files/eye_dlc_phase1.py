import tensorflow as tf
import os
os.environ["DLClight"]="True"
import deeplabcut
import sys

##cat $PBS_GPUFILE
##CUDADEV=$(cat $PBS_GPUFILE | rev | cut -d"u" -f1)
cudadev=os.system("cat $PBS_GPUFILE | rev | cut -d\"u\" -f1")
os.environ['CUDA_VISIBLE_DEVICES'] = str(cudadev)
print(cudadev)
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"

video_file_path = sys.argv[1]
output_file_path = sys.argv[2]
#output_file = video_file_path[:-4] + 'DeepCut_resnet50_universal_eye_trackingApr25shuffle1_969000.h5'
output_file = video_file_path[:-4] + 'DeepCut_resnet50_universal_eye_trackingJul10shuffle1_1030000.h5'

# ### Path to trained model:
#path_config_file = '/allen/programs/braintv/workgroups/cortexmodels/peterl/visual_behavior/DLC_models/universal_eye_tracking-peterl-2019-04-25/config.yaml'
#path_config_file = '/allen/programs/braintv/workgroups/cortexmodels/peterl/visual_behavior/DLC_models/universal_eye_tracking-peterl-2019-07-10/config.yaml'
path_config_file = '/allen/aibs/technology/waynew/eye/universal_eye_tracking-peterl-2019-07-10/config.yaml'

# ### Track points in video and generate h5 file:
deeplabcut.analyze_videos(path_config_file,[video_file_path]) #can take a list of input videos

os.rename(output_file,output_file_path)

