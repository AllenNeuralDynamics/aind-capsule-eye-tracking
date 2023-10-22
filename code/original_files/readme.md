files used by a processing run in lims, plus logs (job is run in 2 phases):
- `.slurm` job scripts submitted to HPC (with paths passed to `.py` files)
- `config.yaml` for DLC project
- `.py` files that import and run DLC (phase 1), fit ellipses (phase 2), and
  write files (both)

the project path used is stored in a data asset **`universal_eye_tracking-peterl-2019-07-10`**:
  - `05529cfc-23fe-4ead-9490-71763e9f7c01` 
  - https://codeocean.allenneuraldynamics.org/data-assets/05529cfc-23fe-4ead-9490-71763e9f7c01/universal_eye_tracking-peterl-2019-07-10