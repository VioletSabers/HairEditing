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
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

#%%
root_path = '/data1/gxy_ctl/ffhq/data_image'

size = 1024
transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
dataset = datasets.ImageFolder(root_path, transform=transform)

#%%
dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=False)

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
#%%
name =dataset.imgs
#%%
with torch.no_grad():
    for i, (data, _) in enumerate(dataloader):
        
        if i < 58240:
            print(f'\r{i}', end='')
            continue
        data = data.cuda()
        img_d = downsample(data)
        # print(img_d.shape)
        print(f"\rloading {i}", end='')
        predictions, HM, FM = get_mask(SegNet, img_d)

        HM = HM.unsqueeze(1)
        FM = FM.unsqueeze(1)
        bg_pre = torch.ones_like(HM) - FM - HM
        mask = torch.cat([HM, torch.zeros_like(HM), FM], dim=1)

        image_utils.writeImageToDisk(
                [mask[0].unsqueeze(0)], [name[i][0].split('/')[-1]], '/data1/gxy_ctl/ffhq/mask'
            )
print('\n')
# %%
