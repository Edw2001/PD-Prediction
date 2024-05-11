# PD-Prediction
## Files
- The input dataset version is in datasets/complete_data(dropbox link).
- The networks folder is defined the network we used.
- save/SupCon stores the running result.
- losses.py defined the supcon loss we used
- main_ce.py is running the network through cross-entropy
- main_supcon.py is running the network through SupCon Loss
- main_linear.py is loading the pre-trained model and giving the classification result.
- evaluuation.ipynb is trying to load the model and evaluate the result.
- baseline_not_pretrained.ipynb is the baseline model.
  
## command: 

## Supcon trained on Cifar10
python main_supcon.py --batch_size 8 --cosine --num_workers 16 --size 128 --epochs 80 --model resnet50 --data_folder "datasets/complete_data" --dataset path --mean "(0.485, 0.456, 0.406)" --learning_rate 0.00001 --std "(0.229, 0.224, 0.225)" --method SupCon --momentum 0.9 --print_freq 1 --save_freq 10

## Cross Entropy trained on Cifar10
python main_ce.py --batch_size 4 --cosine --num_workers 16 --epochs 5 --model resnet50 --dataset "PaHaW" --learning_rate 0.00001 --print_freq 1 --save_freq 10

python main_ce.py --batch_size 8 --cosine --num_workers 16 --epochs 10 --model resnet101 --dataset "PaHaW" --learning_rate 0.5 --print_freq 1 --save_freq 10


## Different parameters
python main_linear.py --batch_size 16 --learning_rate 0.1 --cosine --model resnet101 --save_freq 10 --epochs 100 --dataset "PaHaW" --print_freq 1 --ckpt "save/SupCon/path_models/SupCon_path_resnet101_lr_0.01_decay_0.0001_bsz_16_temp_0.07_trial_0_cosine/last.pth" 

python main_linear.py --batch_size 16 --learning_rate 0.1 --cosine --model resnet50 --save_freq 10 --epochs 100 --dataset "PaHaW" --print_freq 1 --ckpt "save/SupCon/cifar10_models/SupCE_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_128_trial_0_cosine/last.pth" 

python main_linear.py --batch_size 16 --learning_rate 0.1 --cosine --save_freq 10 --epochs 10 --dataset "PaHaW" --print_freq 1 --ckpt "save/SupCon/path_models/SupCon_path_efficientnet_b0_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/ckpt_epoch_500.pth" 
