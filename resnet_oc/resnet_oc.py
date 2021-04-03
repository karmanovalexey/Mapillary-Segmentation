import torchvision

from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from .base_oc_block import BaseOC_Module
from .backbone_utils import IntermediateLayerGetter

def get_resnet34_base_oc_layer3(num_classes=66, pretrained_backbone=False):
    replace_stride_with_dilation = [False, False, False]
    inplanes_scale_factor = 4
    
    print('Using pretrained backbone:', pretrained_backbone)
    backbone = torchvision.models.resnet.resnet34(pretrained=pretrained_backbone,
                                                                 replace_stride_with_dilation=replace_stride_with_dilation)
    
    return_layers = {'layer3': 'out'}
    inplanes = 1024 // inplanes_scale_factor
    outplanes = 512
    
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = ResNet_Base_OC(backbone, inplanes, outplanes, num_classes)
    
    return model


class ResNet_Base_OC(nn.Module):
    def __init__(self, backbone, inplanes, outplanes, num_classes):
        super(ResNet_Base_OC, self).__init__()
        self.backbone = backbone
        self.context = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            BaseOC_Module(in_channels=outplanes, 
                          out_channels=outplanes, 
                          key_channels=outplanes // 2, 
                          value_channels=outplanes // 2, 
                          dropout=0.05, 
                          sizes=([1]))
            )
        self.cls = nn.Conv2d(outplanes, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        
        # print('input: ', x.shape)
        x = self.backbone(x)['out']
        # print('backbone: ', x.shape)
        x = self.context(x)
        # print('context: ', x.shape)
        x = self.cls(x)
        # print('cls: ', x.shape)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        # print('output: ', x.shape)
        
        result = OrderedDict()
        result['out'] = x
        
        return x