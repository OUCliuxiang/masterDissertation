conda activate kd
python train_val_baseline.py --gpus 0 1 --bs 256
python train_val_kd.py --gpus 0 1 --bs 256
python train_val_ac_off_syn.py --gpus 0 1 --bs 256 
python train_val_ac_off_asy.py --gpus 0 1 --bs 256
python train_val_ac_online_syn.py --gpus 0 1 --bs 256
python train_val_ac_online_asy.py --gpus 0 1 --bs 256

