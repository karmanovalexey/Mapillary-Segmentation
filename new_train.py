import wandb
import torch
import os
import time
import glob
import re
import numpy as np

from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from deeplab.deeplab import DeepLab
from unet import UNet

from new_val import val
from utils.mapillary import mapillary
from utils.loss import Loss

NUM_CLASSES = 66

def get_model(model_name, pretrained=False):
    if model_name == 'deeplab':
        return DeepLab(backbone='mobilenet', output_stride=16, num_classes=NUM_CLASSES, sync_bn=False, freeze_bn=False)
    if model_name =='unet':
        return UNet(3,NUM_CLASSES)
    else:
        raise NotImplementedError('Unknown model')

def train(args):
    #Get training data
    assert os.path.exists(args.data_dir), "Error: datadir (dataset directory) could not be loaded"
    dataset_train = mapillary(args.data_dir, 'train', height=args.height, part=1)
    loader = DataLoader(dataset_train, num_workers=4, batch_size=args.batch_size, shuffle=True)
    print('Loaded', len(loader), 'batches')

    model = get_model(args.model, args.pretrained).to(device=args.device)

    criterion = Loss(args)

    savedir = args.save_dir
    savedir = f'./save/{savedir}'

    optimizer = Adam(model.parameters(), 3e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lambda x: (1 - x/args.num_epochs) ** 0.9)

    
    start_epoch = 0
    best_metric = 0
    if args.resume:
        #Must load weights, optimizer, epoch and best value.
        file_resume = savedir + '/{}.pth'.format(args.model)
        
        assert os.path.exists(file_resume), "Error: resume option was used but checkpoint was not found in folder"
        checkpoint = torch.load(file_resume)
        start_epoch = checkpoint['epoch'] + 1
        best_metric = checkpoint['best_iou']
        optimizer.load_state_dict(checkpoint['opt'])
        model.load_state_dict(checkpoint['model'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> Loaded checkpoint at epoch {})".format(checkpoint['epoch']))
    
    for epoch in range(start_epoch, args.num_epochs+1):
        print("----- TRAINING - EPOCH", epoch, "-----")

        patch_loss = []

        model.train()
        for step, (images, labels) in enumerate(tqdm(loader)):

            inputs = images.to(device=args.device)
            targets = labels.to(device=args.device)

            outputs = model(inputs)
            
            optimizer.zero_grad()
            loss = criterion(outputs, targets[:, 0])
            loss.backward()
            optimizer.step()

            patch_loss.append(loss.data.item())
            if (step % 100 == 0) and (step != 0):
                average = np.mean(patch_loss)
                wandb.log({"epoch":epoch, "loss":average, 'lr':scheduler.get_last_lr()[0]}, step=(epoch-1)*18000 + step*args.batch_size)
                patch_loss = []
        
        scheduler.step()

        last_metric = val(args, model, part=0.2)
        print('Val', last_metric)

        if float(last_metric['iou']) > best_metric:
            best_metric = float(last_metric['iou'])
            #if args.epochs_save > 0 and epoch > 0 and epoch % args.epochs_save == 0:
            filename = f'{savedir}/{args.model}.pth'
            torch.save({'model':model.state_dict(), 'opt':optimizer.state_dict(),'scheduler':scheduler.state_dict(), 'epoch':epoch, 'best_iou':best_metric}, filename)
            print(f'save: {filename} (epoch: {epoch})')
    
    return

def main(args):
    savedir = args.save_dir
    savedir = f'./save/{savedir}'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    config = dict(model = args.model,
                    height = args.height,
                    epochs = args.num_epochs,
                    bs = args.batch_size,
                    pretrained = args.pretrained,
                    savedir = args.save_dir)
    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    
    log_mode = 'online' if args.wandb else 'disabled'
    with wandb.init(project=args.project_name, config=config, mode=log_mode, save_code=True):
        print('Run properties:', config)
        print("========== TRAINING ===========")
        train(args)
        print("========== TRAINING FINISHED ===========")

if __name__== '__main__':
    wandb.login()
    parser = ArgumentParser()
    
    parser.add_argument('--data-dir', required=True, help='Mapillary directory')
    parser.add_argument('--model', required=True, choices=['deeplab', 'unet'], help='Tell me what to train')
    parser.add_argument('--loss', default='BCE', help='Loss name, either BCE or Focal')
    parser.add_argument('--height', type=int, default=600, help='Height of images, nothing to add')
    parser.add_argument('--num-epochs', type=int, default=10, help='If you use resume, give a number considering for how long it trained')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--save-dir', help='Where to save your model')
    parser.add_argument('--pretrained', action='store_true', help='Whether to use pretrained backbone')
    parser.add_argument('--resume', action='store_true', help='Resumes from the last save from --savedir directory')
    parser.add_argument('--wandb', action='store_true', help='Whether to log metrics to wandb')    
    parser.add_argument('--project-name', default='Junk', help='Project name for weights and Biases')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--epochs-save', type=int, default=3, help='You can use this value to save model every X epochs')
    main(parser.parse_args())
