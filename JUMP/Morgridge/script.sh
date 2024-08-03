export PYTHONPATH=.
export WANDB_API_KEY= #YOUR API KEY
export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nproc_per_node=4 dinov2/train/train.py --config-file dinov2/configs/train/vits16_jump.yaml --output-dir ./test/ --experiment test train.dataset_path=JUMPDataset:root=/scr/apillai/modelTraining/sample100k:metadata_path=/scr/apillai/modelTraining/sample100k/jump_devset_training_sample.csv
