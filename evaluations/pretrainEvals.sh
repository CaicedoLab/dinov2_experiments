#copy large files from /staging
cp /staging/groups/caicedo_group/imagenet.zip ./ 
cp /staging/groups/caicedo_group/extra.zip ./ 

#extracting images from files
unzip imagenet.zip &
unzip extra.zip &

wait

#setting up train and val directory
mkdir ./imagenet/train/train
cd ./imagenet/train
mv * train
cd ..
cd ..

mkdir ./imagenet/val/val
cd ./imagenet/val
mv * val
cd ..
cd ..

#setting directories
mkdir /home/dinov2/checkpoints
cp dinov2_vits14_reg4_pretrain.pth /home/dinov2/checkpoints
cp knn.py /home/dinov2/dinov2/eval
export jobDir=$(pwd)

cd /home/dinov2
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0

mkdir output

#running pretrained evals (python3)
#torchrun --nproc_per_node=1 ./dinov2/run/eval/knn.py \
  #  --config-file ./dinov2/configs/eval/vits14_reg4_pretrain.yaml \
  #  --pretrained-weights ./checkpoints/dinov2_vits14_reg4_pretrain.pth \
  #  --output-dir ./output \
  #  --train-dataset ImageNet:split=TRAIN:root=$jobDir/imagenet/train:extra=$jobDir/extra \
 #   --val-dataset ImageNet:split=VAL:root=$jobDir/imagenet/val:extra=$jobDir/extra &

python -m torch.distributed.launch --nproc_per_node=1 ./dinov2/eval/knn.py --config-file=./dinov2/configs/eval/vits14_reg4_pretrain.yaml --pretrained-weights=./checkpoints/dinov2_vits14_reg4_pretrain.pth --output-dir=./output --train-dataset=ImageNet:split=TRAIN:root=$jobDir/imagenet/train:extra=$jobDir/extra --val-dataset=ImageNet:split=VAL:root=$jobDir/imagenet/val:extra=$jobDir/extra &

wait
    
#compress file for output transfer
tar -cvzf pretrainEvalsOutput.tar.gz ./output

cp pretrainEvalsOutput.tar.gz $jobDir

