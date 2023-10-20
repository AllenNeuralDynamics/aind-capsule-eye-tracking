#!/bin/bash
#SBATCH --partition=braintv
#SBATCH --qos=production
#SBATCH --nodes=1 --cpus-per-task=1 --mem=20G
#SBATCH --time=5:00:00
#SBATCH --export=NONE
#SBATCH --job-name=ECEPHYS_EYE_ELLIPSE_FIT_QUEUE_1298465622
#SBATCH --output=/allen/programs/mindscope/production/openscope/prod0/specimen_1292228830/ecephys_session_1298465622/202310191437_ECEPHYS_EYE_ELLIPSE_FIT_QUEUE_1298465622.log
#SBATCH --chdir=/allen/programs/mindscope/production/openscope/prod0/specimen_1292228830/ecephys_session_1298465622/
#SBATCH --mail-type=NONE 
umask 022
/allen/aibs/technology/conda/run_miniconda.sh /allen/aibs/technology/waynew/conda/dlcPy36 python /allen/aibs/technology/waynew/eye/bin/eye_dlc_phase2.py /allen/programs/mindscope/production/openscope/prod0/specimen_1292228830/ecephys_session_1298465622/eye_tracking/1298465622_692072_20230921.eyeDeepCut_resnet50_universal_eye_trackingJul10shuffle1_1030000.h5 /allen/programs/mindscope/production/openscope/prod0/specimen_1292228830/ecephys_session_1298465622/eye_tracking/1298465622_ellipse.h5
rtn_code=$?
curl --request GET -d return_code=$rtn_code http://lims2/jobs/finish_job/1304895964 --max-time 300 --retry 5 --retry-delay 300
exit $rtn_code
