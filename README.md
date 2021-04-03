# Mapillary_semantics
This repository uses deeplabv3+, unet and erfnet to train on mapillary. It uses weights &amp; biases


python val.py --model deeplab --model-dir deeplab_main_run --height 1080 --project-name "Evaluation" --data-dir /media/alexey/HDD/Mapillary --batch-size 1


python train.py --model unet --save-dir unet_main_run --height 1080 --project-name "UNet training" --data-dir /media/alexey/HDD/Mapillary/ --batch-size 1 --resume --num-epochs 25
