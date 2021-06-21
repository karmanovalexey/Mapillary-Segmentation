import numpy as np
import os
import random

from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad, InterpolationMode
from torchvision.transforms import ToTensor, ToPILImage
from .transform import Relabel, ToLabel, Colorize

from PIL import Image, ImageOps

from torch.utils.data import Dataset

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return filename.endswith(".jpg")

def is_label(filename):
    return filename.endswith(".png")

def image_path(root, name):
    return os.path.join(root, f'{name}')


class MyCoTransform(object):
    def __init__(self, augment=True, height=600):
        self.augment = augment
        self.height = height
        pass
    def __call__(self, input, target):
        # do something to both images
        if self.height == 1080:
            input =  Resize((1080,1920), InterpolationMode.BILINEAR)(input)
            target = Resize((1080,1920), InterpolationMode.NEAREST)(target)
        else:
            input =  Resize((self.height,self.height*2), InterpolationMode.BILINEAR)(input)
            target = Resize((self.height,self.height*2), InterpolationMode.NEAREST)(target)

        if(self.augment):
            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
            
            #Random translation 0-2 pixels (fill rest with padding
            transX = random.randint(-2, 2) 
            transY = random.randint(-2, 2)

            input = ImageOps.expand(input, border=(transX,transY,0,0), fill=0)
            target = ImageOps.expand(target, border=(transX,transY,0,0), fill=255) #pad label filling with 255
            input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
            target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))   

        input = ToTensor()(input)
        input = Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])(input)
        target = ToLabel()(target)
        target = Relabel(255, 65)(target)

        return input, target

class mapillary(Dataset):

    def __init__(self, root, subset='train', height=600, part=1.):
        self.images_root = os.path.join(root, subset)
        self.labels_root = os.path.join(root, subset)
        
        self.images_root += '/1920_1080/images'
        self.labels_root += '/1920_1080/labels'

        print (self.images_root)
        
        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()
        self.filenames = self.filenames[:int(part*len(self.filenames))]

        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        self.filenamesGt.sort()
        self.filenamesGt = self.filenamesGt[:int(part*len(self.filenamesGt))]
        
        if subset=='train':
            self.co_transform = MyCoTransform(augment=True, height=height)
        elif subset=='val':
            self.co_transform = MyCoTransform(augment=False, height=height)


    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(image_path(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')

        image, label = self.co_transform(input=image, target=label)

        return image, label

    def __len__(self):
        return len(self.filenames)

