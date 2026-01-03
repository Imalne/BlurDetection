python train.py 2 >&1 | tee ./logs/train_log.txt
python train_w_infonce.py 2 >&1 | tee ./logs_infonce/train_log.txt
