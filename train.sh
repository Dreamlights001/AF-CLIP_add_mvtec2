
# train for zero-shot model
data_dir=/home/wang/datasets
CUDA_VISIBLE_DEVICES=0 python main.py --log_dir ./train_log --dataset mvtec --data_dir $data_dir
CUDA_VISIBLE_DEVICES=0 python main.py --log_dir ./train_log --dataset visa --test_dataset mvtec --data_dir $data_dir


