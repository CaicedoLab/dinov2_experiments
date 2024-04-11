#script file to train dinov2 on a non-slurm environment (useful if trying to run on CHTC)

#copy large files from /staging
cp /staging/groups/caicedo_group/imagenet.zip ./
cp /staging/groups/caicedo_group/extra.zip ./

#extracting images from files
unzip imagenet.zip &
unzip extra.zip &

wait

#setup train dir
mkdir ./imagenet/train/train
cd ./imagenet/train
mv * train
cd ..
cd ..

#move config and modified train file
mv config.yaml /home/dinov2/dinov2/configs/train
mv train.py /home/dinov2/dinov2/train

#change to dinov2 dir
export jobDir=$(pwd)
cd /home/dinov2
mkdir output

#init PYTHONPATH, CUDA
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0

#training script
python -m torch.distributed.launch --nproc_per_node=1 ./dinov2/train/train.py --config-file=./dinov2/configs/train/config.yaml --output-dir=./output --no-resume train.dataset_path=ImageNet:split=TRAIN:root=$jobDir/imagenet/train:extra=$jobDir/extra & 

wait

#copy output back to job directory
tar -cvzf TrainingOutput.tar.gz output
cp TrainingOutput.tar.gz $jobDir
