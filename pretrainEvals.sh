echo "var values before modifying:"
echo $CUDA_VISIBLE_DEVICES
echo $NVIDIA_VISIBLE_DEVICES

#copy large files from /staging
cp /staging/groups/caicedo_group/imagenet.zip ./
cp /staging/groups/caicedo_group/extra.zip ./

#extracting images from files
unzip imagenet.zip
unzip extra.zip

#setting up train directory correctly?
mkdir ./imagenet/train/train
cd ./imagenet/train
mv * train
cd ..
cd ..

#setting directories
mkdir /home/dinov2/checkpoints
cp dinov2_vits14_reg4_pretrain.pth /home/dinov2/checkpoints
export jobDir=$(pwd)
cd /home/dinov2

mkdir output
mkdir ./output/knn

#init PYTHONPATH
export PYTHONPATH=/home/dinov2
export CUDA_VISIBLE_DEVICES=0

#running pretrained evals (python3)
torchrun --nproc_per_node=1 ./dinov2/run/eval/knn.py \
    --config-file ./dinov2/configs/eval/vits14_reg4_pretrain.yaml \
    --pretrained-weights ./checkpoints/dinov2_vits14_reg4_pretrain.pth \
    --output-dir ./output/knn \
    --train-dataset ImageNet:split=TRAIN:root=$jobDir/imagenet/train:extra=$jobDir/extra \
    --val-dataset ImageNet:split=VAL:root=$jobDir/imagenet/val:extra=$jobDir/extra

#compress file for output transfer
tar -cvzf pretrainEvalsOutput.tar.gz ./output

cp pretrainEvalsOutput.tar.gz $jobDir

#copy output to staging and setup output in staging + analysis purposes
cp pretrainEvalsOutput.tar.gz /staging/groups/caicedo_group/aditya_pillai
cd /staging/groups/caicedo_group/aditya_pillai
tar -xvf pretrainEvalsOutput.tar.gz -C ./alltrainings/job3
