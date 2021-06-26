import os
import numpy as np
import torch
import wandb
from PIL import Image

from tqdm import tqdm
from time import perf_counter
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad, InterpolationMode, ToTensor, ToPILImage
from deeplab.deeplab import DeepLab


from utils.mapillary import mapillary
from utils.transform import Relabel, ToLabel, Colorize
from utils.iouEval import iouEval
from utils.loss import Loss
from utils.mapillary_pallete import MAPILLARY_CLASSNAMES as classnames

NUM_CLASSES = 66

def get_model(model_name, pretrained=False):
    if model_name == 'deeplab':
        return DeepLab(backbone='resnet', output_stride=16, num_classes=NUM_CLASSES, sync_bn=False, freeze_bn=False)
    else:
        raise NotImplementedError('Unknown model')

def load_checkpoint(model_path):
    #Must load weights, optimizer, epoch and best value.
    file_resume = f'{model_path}'
    #file_resume = savedir + '/model-{}.pth'.format(get_last_state(savedir))
    assert os.path.exists(file_resume), "No model checkpoint found"
    checkpoint = torch.load(file_resume)

    return checkpoint

def get_example(datadir, height):
    with open(datadir + '/val/1920_1080/images/_4SjTmQ-zn3XSv4D1-Tg4w.jpg', 'rb') as f:
        image = Image.open(f).convert('RGB')
    with open(datadir + '/val/1920_1080/labels/_4SjTmQ-zn3XSv4D1-Tg4w.png', 'rb') as f:
        label = Image.open(f).convert('P')

    input =  Resize((height,height*2), InterpolationMode.BILINEAR)(image)
    target = Resize((height,height*2), InterpolationMode.NEAREST)(label)

    input = ToTensor()(input)
    input = Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])(input)
    target = ToLabel()(target)
    target = Relabel(255, 65)(target)

    return input, target

def val(args, model, part=1.,):
    dataset_val = mapillary(args.data_dir, 'val', height=args.height, part=part) # Taking only 10% of images
    loader_val = DataLoader(dataset_val, num_workers=4, batch_size=1, shuffle=False)
    print('Loaded', len(loader_val), 'examples')

    val_loss = []
    time_val = []

    criterion = Loss(args)
    model.eval()
    iouEvalVal = iouEval(NUM_CLASSES, args.device)
    color_transform = Colorize(NUM_CLASSES)
    
    with torch.no_grad():
        for step, (images, labels) in enumerate(tqdm(loader_val)):

            images = images.to(device=args.device)
            labels = labels.to(device=args.device)

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

        data = [[label, val] for (label, val) in zip(classnames[:-1], iou_classes.tolist())]
        table = wandb.Table(data=data, columns = ["label", "value"])
        wandb.log({"Predictions" : wandb.plot.bar(table, "label", "value", title="Predictions")})


        main_classes = np.mean([iou_classes[2],iou_classes[3],iou_classes[11],iou_classes[13],iou_classes[15],iou_classes[17],
                            iou_classes[19],iou_classes[23],iou_classes[24],iou_classes[29],iou_classes[30],iou_classes[31],
                            iou_classes[48],iou_classes[50],iou_classes[54],iou_classes[55]])

        wandb.log({'val_fps':1./np.mean(time_val),
        'val_IOU':iouVal,
        'val_loss':np.mean(val_loss),
        'main_classes': main_classes})

        img_ex, lab_ex = get_example(args.data_dir, args.height)
        img_ex, lab_ex = img_ex.to(device=args.device), lab_ex.to(device=args.device)
        outputs = model(img_ex.unsqueeze(0))

        examples = [np.moveaxis(np.array(color_transform(outputs[0].cpu().max(0)[1].data.unsqueeze(0))),0,2),
                np.moveaxis(np.array(color_transform(lab_ex.cpu().data)),0,2),]
        wandb.log({args.model:[wandb.Image(i) for i in examples]})


    return {'iou':iouVal.data, 'fps':1./np.mean(time_val), 'mean_time':np.mean(time_val)}

def main(args):
    config = dict(model = args.model,
                    height = args.height,
                    bs = args.batch_size,
                    mode = 'Validation')
    
    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    
    log_mode = 'online' if args.wandb else 'disabled'
    with wandb.init(project=args.project_name, config=config, mode=log_mode):
        print('Run properties:', config)
        model = get_model(args.model, False).to(device=args.device)

        checkpoint = load_checkpoint(args.model_path)
        model.load_state_dict(checkpoint['model'])

        print("========== VALIDATING ===========")
        print(val(args, model, part=0.5))
        print("========== VALIDATING FINISHED ===========")

if __name__ == '__main__':
    wandb.login()
    parser = ArgumentParser()
    parser.add_argument('--data-dir', help='Mapillary directory')
    parser.add_argument('--model', choices=['deeplab'], help='Tell me what to train')
    parser.add_argument('--loss', default='BCE', help='Loss name, either BCE or Focal')
    parser.add_argument('--height', type=int, default=1080, help='Height of images, nothing to add')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--model-path', required=True, help='Where to load your model from')
    parser.add_argument('--wandb', action='store_true', help='Whether to log metrics to wandb')    
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--project-name', default='OC Results', help='Project name for weights and Biases')
    main(parser.parse_args())
