docker run \
            --gpus all -it \
            --rm \
            --ipc=host \
            -v /home/adeshkin/projects/tools/Mapillary:/workspace/Mapillary \
            -v /home/karmanov_aa/Mapillary-Segmentation:/workspace/MS \
            -v /home/karmanov_aa/Resnet_OC:/workspace/Resnet_OC \
            -v /home/karmanov_aa/taganrog:/workspace/video \
            resnet_oc
