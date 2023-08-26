import os

string = "python train.py --name cub200_train_mix --config_file configs/cub200_mix.yaml --num_node 1 --tensorboard --load_path OUTPUT/pretrained_model/CC_pretrained.pth"

os.system(string)

