# Running dinov2 JUMP experiments on CHTC & Morgridge GPU servers
Run dinov2 model with the following steps. Training metrics can be viewed on weights and biases (https://wandb.ai/site)

## CHTC (running through Docker) :
### 1. Run `condor_submit training.sub` to submit a job to a cluster. Both neccessary files found in `./CHTC`
  - Ensure dinov2 repository exists in docker container or some directory that is accessible
  - Add your own wandb api key to `training.sh` if you would like to track training metrics
  - Files and directories can be stored persistently in the `/scratch` directory within the job. (Recommended for storing datasets). Several     sample JUMP datasets are currently available in /scratch/appillai directory


  

## GPU servers (running directly):
### 1. Have dinov2 repository ready 
### 2. Install dinov2 dependencies in a conda based environment. 
> NOTE: Certain package versions are not available in the provided conda channels. It might be easier to install all packages in the base environment
### 3. Run `script.sh` to start training. Found in `./Morgridge`
  - Choose GPUs through `CUDA_VISIBLE_DEVICES` env variable
  - JUMP datasets available in `/mnt/cephfs/mir/jcaicedo/cellpainting-data/sampleDatasets/'

