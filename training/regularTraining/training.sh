#copy large files from /staging
#cp /staging/groups/caicedo_group/imagenet.zip ./
#cp /staging/groups/caicedo_group/extra.zip ./

#extracting images from files
#unzip imagenet.zip &
#unzip extra.zip &

#wait

#setup train dir
#mkdir ./imagenet/train/train
#cd ./imagenet/train
#mv * train
#cd ..
#cd ..

mv config.yaml /home/wandbIntegrate/dinov2/configs/train
mv train.py /home/wandbIntegrate/dinov2/train
mv __init__.py /home/wandbIntegrate/dinov2/fsdp

#change to dinov2 dir
export jobDir=$(pwd)
cd /home/wandbIntegrate
mkdir output

#init PYTHONPATH
export PYTHONPATH=.
#export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=61bf6d44e974b73170041842d514538f0ab4ed00

#training script. initally starting with 1 node 1 epoch (torchrun --nproc_per_node=1)
#torchrun --nproc_per_node=1 ./dinov2/run/train/train.py \
 #   --nodes 1 \
 #   --config-file dinov2/configs/train/config.yaml \
 #   --output-dir ./output \
 #   --no-resume \
 #   train.dataset_path=ImageNet:split=TRAIN:root=$jobDir/imagenet/train:extra=$jobDir/extra &

#python -m torch.distributed.launch
torchrun --nproc_per_node=1 ./dinov2/train/train.py --config-file=./dinov2/configs/train/config.yaml --output-dir=./output train.dataset_path=ImageNet:split=TRAIN:root=/scratch/appillai/imagenet/train:extra=/scratch/appillai/extra &
wait

tar -cvzf TrainingOutput.tar.gz output
cp TrainingOutput.tar.gz $jobDir
