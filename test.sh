# store weight in the dir ./weight like mvtec_prompt.pt, mvtec_adaptor.pt which are trained on mvtec dataset
# and visa_prompt.pt, visa_adaptor.pt which are trained on visa dataset
# zero-shot
data_dir=/home/wang/datasets
#CUDA_VISIBLE_DEVICES=0 python main.py --log_dir ./log/zero-shot --dataset visa --test_dataset mvtec --weight ./weight --data_dir $data_dir
CUDA_VISIBLE_DEVICES=0 python main.py --log_dir ./log/zero-shot --dataset mvtec --weight ./weight --data_dir $data_dir

# few-shot


