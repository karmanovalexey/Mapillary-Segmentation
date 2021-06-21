#!/bin/sh
python3 new_train.py --data-dir /workspace/Mapillary  --model deeplab --loss Focal --height 1080 --num-epochs 100 --batch-size 2 --pretrained --project-name DeepLab --save-dir DeepLab