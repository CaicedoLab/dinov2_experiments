# Running experiments on CHTC & Morgridge GPU servers
Run dinov2 model with the following steps. Training metrics can be viewed on weights and biases (https://wandb.ai/site)

## CHTC (running through Docker) :
### 1. Run `condor_submit training.sub` to submit a job to a cluster. Both neccessary files found in `./CHTC`
  - Add wandb api key to `training.sh` if you would like to track training metrics
  - Be sure to update the docker image for your own use case
  - Files and directories can be stored persistently in the `/scratch` directory within the job. (Recommended for storing datasets)

  

## GPU servers (running directly):
### 1. Have dinov2 repository ready 
### 2. Install dinov2 dependencies in a conda based environment. 
> NOTE: Certain package versions are not available in the provided conda channels. It might be easier to install all packages in the base environment
### 3. Run `script.sh` to start training
  - Choose GPUs through `CUDA_VISIBLE_DEVICES` env variable

