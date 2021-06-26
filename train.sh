#!/bin/sh
python3 new_train.py --data-dir /workspace/Mapillary --resume --model unet --loss Focal --height 1080 --wandb --num-epochs 100 --batch-size 4 --pretrained --project-name DeepLab --save-dir DeepLab
