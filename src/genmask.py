#%%
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image
from torchvision import datasets, transforms
import torch.nn as nn


import sys
sys.path.append('./')
from networks import deeplab_xception_transfer
from networks.graphonomy_inference import get_mask
import torch.nn.functional as F
from utils import image_utils

#%%

def gen(root_path, mask_path):
    
    size = 1024
    transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    img = Image.open(root_path).convert("RGB")
    data = transform(img)
    SegNet = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(
            n_classes=20,
            hidden_layers=128,
            source_classes=7,
        )
    state_dict = torch.load('./pretrain_model/inference.pth')
    SegNet.load_source_model(state_dict)
    SegNet = SegNet.cuda()
    SegNet.eval()

    downsample = nn.Upsample(scale_factor=256/1024, mode='bilinear').cuda()
    cnt = 0
    
    data = data.unsqueeze(0).cuda()
    img_d = downsample(data)
    predictions, HM, FM = get_mask(SegNet, img_d)

    HM = HM.unsqueeze(1)
    FM = FM.unsqueeze(1)
    bg_pre = torch.ones_like(HM) - FM - HM
    mask = torch.cat([HM, torch.zeros_like(HM), FM], dim=1)
    name = root_path.split('/')[-1].split('.')[0]
    image_utils.writeImageToDisk(
            [mask[0].unsqueeze(0)], [name + '.png'], mask_path
        )
    print('genmask ' + root_path.split('/')[-1] + ' ok')

if __name__ == '__main__':
    image = 'dk.jpg'
    gen('./data/images/' + image, './data/masks')