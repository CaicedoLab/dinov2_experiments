#update repo in docker with new repo (optional)
#cp -r /PATH/TO/UPDATED/REPO /home/dino

cd /home/dino/DINOv2_CellPainting/dinov2
mkdir test

#set env variables
export PYTHONPATH=. 
export WANDB_API_KEY= #YOUR WANDB API KEY 

#run training. Set gpus, provide config file, path to dataset
torchrun --nproc_per_node=1 ./dinov2/train/train.py --config-file ./dinov2/configs/train/vits16_jump.yaml --output-dir ./test/ --experiment test train.dataset_path=JUMPDataset:root=/scratch/appillai/sampleDevset/:metadata_path=/scratch/appillai/sampleDevset/jump_devset_training_sample.csv

#convert output dir to tar.gz -- Make sure this file is transferred as output in chtc job submit file (.sub)
tar -cvzf 4gpuOutput.tar.gz test

