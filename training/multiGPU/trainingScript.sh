#### script.sh #####
#!/bin/sh

# untar the test and training data
#tar zxf MNIST_data.tar.gz

if [ "$1" = "rank0" ]
then
      # rank0 port
       h=$(hostname)
   p=$2
        echo "$h $p" > contact_file
      python3 htchirp.py put contact_file rank0_contact
      rm contact_file
else
      h=$1
   p=$2
fi

python3 htchirp.py ulog "Running using $h:$p for the rendezvous"
python3 htchirp.py set_job_attr C10dEndpoint \"$h:$p\"
python3 htchirp.py set_job_attr ProvisionerState 2

mv config.yaml /home/wandbIntegrate/dinov2/configs/train
mv train.py /home/wandbIntegrate/dinov2/train
mv __init__.py /home/wandbIntegrate/dinov2/fsdp

#change to dinov2 dir
export jobDir=$(pwd)
cd /home/wandbIntegrate
mkdir output

export PYTHONPATH=.
export WANDB_API_KEY=#your WANDB api key

torchrun --nnodes 1:4 --nproc_per_node 2 --rdzv_backend c10d --rdzv-id 1 --rdzv-endpoint "$h:$p" ./dinov2/train/train.py --config-file=./dinov2/configs/train/config.yaml --output-dir=./output train.dataset_path=ImageNet:split=TRAIN:root=/scratch/appillai/imagenet/train:extra=/scratch/extra
