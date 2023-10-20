#!/bin/bash
#SBATCH --partition=braintv
#SBATCH --qos=production
#SBATCH --nodes=1 --cpus-per-task=4 --gpus=1 --mem=32G
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --job-name=ECEPHYS_EYE_TRACKING_QUEUE_1298465622
#SBATCH --output=/allen/programs/mindscope/production/openscope/prod0/specimen_1292228830/ecephys_session_1298465622/202310191112_ECEPHYS_EYE_TRACKING_QUEUE_1298465622.log
#SBATCH --chdir=/allen/programs/mindscope/production/openscope/prod0/specimen_1292228830/ecephys_session_1298465622/
#SBATCH --mail-type=NONE 
umask 022
/allen/aibs/technology/conda/run_miniconda.sh /allen/aibs/technology/waynew/conda/dlcPy36 python /allen/aibs/technology/waynew/eye/bin/eye_dlc_phase1.py /allen/programs/mindscope/production/openscope/prod0/specimen_1292228830/ecephys_session_1298465622/1304435871/1298465622_692072_20230921.eye.mp4 /allen/programs/mindscope/production/openscope/prod0/specimen_1292228830/ecephys_session_1298465622/eye_tracking/1298465622_692072_20230921.eyeDeepCut_resnet50_universal_eye_trackingJul10shuffle1_1030000.h5
rtn_code=$?
curl --request GET -d return_code=$rtn_code http://lims2/jobs/finish_job/1304846491 --max-time 300 --retry 5 --retry-delay 300
exit $rtn_code
