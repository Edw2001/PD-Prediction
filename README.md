# PD-Prediction

command: python main_supcon.py --batch_size 16 --cosine --num_workers 16 --size 128 --epochs 100 --model resnet101 --data_folder "D:\Cornell Tech\2024 Spring\ML for health\PD-Prediction\PD-Prediction\complete_data" --dataset path --mean "(0.485, 0.456, 0.406)" --learning_rate 0.01 --std "(0.229, 0.224, 0.225)" --method SupCon --momentum 0.9 --print_freq 1 --save_freq 10


python main_linear.py --batch_size 16 --num_workers 4 --epochs 50 --model resnet101 --data_folder "D:\Cornell Tech\2024 Spring\ML for health\PD-Prediction\PD-Prediction\complete_data" --dataset path --learning_rate 0.01 --ckpt "save/SupCon/path_models/SupCon_path_resnet101_lr_0.01_decay_0.0001_bsz_16_temp_0.07_trial_0_cosine/last.pth" --mean "(0.485, 0.456, 0.406)" --std "(0.229, 0.224, 0.225)"
