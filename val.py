import os
import re
import random
import time
import numpy as np
import torch
import math
import glob
from time import perf_counter

from erfnet import ErfNet
from unet import UNet
from deeplab.deeplab import DeepLab
from resnet_oc.resnet_oc import get_resnet34_base_oc_layer3

from mapillary import mapillary
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from transform import Colorize

from iouEval import iouEval
import wandb

NUM_CLASSES = 66

def load_checkpoint(model_path):
    #Must load weights, optimizer, epoch and best value.
    file_resume = f'{model_path}'
    #file_resume = savedir + '/model-{}.pth'.format(get_last_state(savedir))
    assert os.path.exists(file_resume), "No model checkpoint found"
    checkpoint = torch.load(file_resume)

    return checkpoint

class CrossEntropyLoss2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.NLLLoss()

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)


def val(args, model, part=1., mode='val'):
    dataset_val = mapillary(args.data_dir, 'val', height=args.height, part=part) # Taking only 10% of images
    loader_val = DataLoader(dataset_val, num_workers=4, batch_size=args.batch_size, shuffle=False)
    print('Loaded', len(loader_val), 'files')

    val_loss = []
    time_val = []
    val_iou = []

    criterion = CrossEntropyLoss2d()
    model.eval()
    iouEvalVal = iouEval(NUM_CLASSES)
    color_transform = Colorize(NUM_CLASSES)
    
    with torch.no_grad():
        for step, (images, labels) in enumerate(loader_val):

            images = images.cuda()
            labels = labels.cuda()

            torch.cuda.synchronize()
            t1 = perf_counter()

            outputs = model(images)

            torch.cuda.synchronize()
            t2 = perf_counter()

            loss = criterion(outputs, labels[:, 0])

            val_loss.append(loss.data.item())
            time_val.append((t2 - t1)/images.shape[0]) #time

            iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, labels.data) #IOU
            iouVal, iou_classes = iouEvalVal.getIoU()
            val_iou.append(iouVal)

        wandb.log({'val_fps':1./np.mean(time_val),
        'val_IOU':np.mean(val_iou),
        'val_loss':np.mean(val_loss)})

        examples = [np.moveaxis(np.array(color_transform(outputs[0].cpu().max(0)[1].data.unsqueeze(0))),0,2),
                np.moveaxis(np.array(color_transform(labels[0].cpu().data)),0,2)]
        wandb.log({args.model:[wandb.Image(i) for i in examples]})

    return [np.mean(val_iou), 1./np.mean(time_val), np.mean(val_loss)]


def main(args):

    config = dict(model=args.model, dataset='Mapillary', mode='Validation')
    with wandb.init(project=args.project_name, config=config):
        print('Using', args.model)
        if args.model == 'erfnet':
            model = ErfNet(NUM_CLASSES)
        elif args.model == 'unet':
            model = UNet(3,NUM_CLASSES)
        elif args.model == 'deeplab':
            model = DeepLab(backbone='mobilenet', output_stride=16, num_classes=NUM_CLASSES, sync_bn=False, freeze_bn=False)
        elif args.model == 'resnet_oc':
            model = get_resnet34_base_oc_layer3(pretrained_backbone=True)
        else:
            raise NotImplementedError('Unknown model')
        model = torch.nn.DataParallel(model).cuda()

        checkpoint = load_checkpoint(args.model_path)
        model.load_state_dict(checkpoint['model'])
        print("=> Loaded checkpoint at epoch {}".format(checkpoint['epoch']))

        print("========== VALIDATING ===========")
        val(args, model, part=0.5)
        print("========== VALIDATING FINISHED ===========")

if __name__ == '__main__':
    wandb.login()
    parser = ArgumentParser()
    parser.add_argument('--data-dir', help='Mapillary directory')
    parser.add_argument('--model', choices=['erfnet', 'unet', 'deeplab', 'resnet_oc'], help='Tell me what to train')
    parser.add_argument('--height', type=int, default=1080, help='Height of images, nothing to add')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--model-path', required=True, help='Where to load your model from')
    parser.add_argument('--project-name', default='Evaluation', help='Project name for weights and Biases')
    main(parser.parse_args())
