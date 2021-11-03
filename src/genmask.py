#%%
import sys
sys.path.append('./')
import torch
import os
from PIL import Image
from torchvision import datasets, transforms
import torch.nn as nn
from utils_c import image_utils as ImgU


import sys
sys.path.append('./')
from networks import deeplab_xception_transfer
from networks.graphonomy_inference import get_mask
import torch.nn.functional as F
from utils_c import image_utils
import numpy as np

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
    predictions, HM, FM, BG = get_mask(SegNet, img_d)

    HM = HM.unsqueeze(1)
    FM = FM.unsqueeze(1)
    BG = BG.unsqueeze(1)
    mask = torch.cat([BG, FM, HM], dim=1)
    name = root_path.split('/')[-1].split('.')[0]
    image_utils.writeImageToDisk(
            [mask[0].unsqueeze(0)], [name + '.png'], mask_path
        )
    print('genmask ' + root_path.split('/')[-1] + ' ok')

def gen_fullhair(image_hair, mask_hair):
    assert(image_hair.shape == (1, 3, 1024, 1024))
    assert(mask_hair.shape == (1, 1, 1024, 1024))

    with torch.no_grad():
        mask_hair = ImgU.erosion(mask_hair, iteration=5)

        label = torch.tensor(range(512)).cuda()

        for i in range(1024):
            if mask_hair[0, 0, i, :512].sum() == 0:
                continue
            L = label[(mask_hair[0, 0, i, :512] * label) != 0].min()
            R = label[(mask_hair[0, 0, i, :512] * label) != 0].max() + 1
            if (R == 512):
                continue
            select = image_hair[0, :, i, L:R]
            select = select.unsqueeze(1).repeat(1, 512, 1).reshape(3, 512*(R-L))
            mask_hair[0, 0, i, L:512] = 1
            image_hair[0, :, i, L:512] = select[:,:512-L]
        
        label = label + 512
        for i in range(1024):
            if mask_hair[0, 0, i, 512:].sum() == 0:
                continue
            L = label[(mask_hair[0, 0, i, 512:] * label) != 0].min()
            R = label[(mask_hair[0, 0, i, 512:] * label) != 0].max() + 1
            if (L == 512):
                continue
            select = image_hair[0, :, i, L:R]
            select = select.unsqueeze(1).repeat(1, 512, 1).reshape(3, 512*(R-L))
            mask_hair[0, 0, i, 512:L+1] = 1
            image_hair[0, :, i, 512:L+1] = select[:,-(L+1-512):]
    return image_hair, mask_hair

if __name__ == '__main__':
    image = '00091.png'
    mask_path = 'data/masks/'
    image_path = 'data/hair/'
    size = 1024
    transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    img = Image.open(image_path + image).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0)

    mask = Image.open(mask_path + image).convert("RGB")
    mask = transform(mask)
    mask_hair = mask[2, :, :].unsqueeze(0).unsqueeze(0)

    gen_fullhair(img, mask_hair)

