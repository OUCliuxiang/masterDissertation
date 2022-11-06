conda activate kd
python train_val_baseline.py --gpus 0  --bs 128 --epochs 200 
python train_val_kd.py --gpus 0  --bs 128 --epochs 200

