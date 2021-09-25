#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torchvision.transforms import transforms
import pandas as pd

import sys
sys.path.append('./')
from utils import optimizer_utils, image_utils
from utils.config import cfg

# parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
# parser.add_argument('--image_path', type=str, default='./data/images/67172.jpg', help='Path to image')
# parser.add_argument('--hairmask_path',type=str, default='./data/masks/67172.png', help='Path to hair mask')
# parser.add_argument('--orientation_root', type=str, default='./orient', help='Root to save hair orientation map')



def DoG_fn(kernel_size, channel_in, channel_out, theta):
    # params
    sigma_h = nn.Parameter(torch.ones(channel_out) * 1.0, requires_grad=False).cuda()
    sigma_l = nn.Parameter(torch.ones(channel_out) * 2.0, requires_grad=False).cuda()
    sigma_y = nn.Parameter(torch.ones(channel_out) * 2.0, requires_grad=False).cuda()

    # Bounding box
    xmax = kernel_size // 2
    ymax = kernel_size // 2
    xmin = -xmax
    ymin = -ymax
    ksize = xmax - xmin + 1
    y_0 = torch.arange(ymin, ymax+1).cuda()
    y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1).float()
    x_0 = torch.arange(xmin, xmax+1).cuda()
    x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize).float()   # [channel_out, channelin, kernel, kernel]

    # Rotation
    # don't need to expand, use broadcasting, [64, 1, 1, 1] + [64, 3, 7, 7]
    x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
    y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))

    gb = (torch.exp(-.5 * (x_theta ** 2 / sigma_h.view(-1, 1, 1, 1) ** 2 + y_theta ** 2 / sigma_y.view(-1, 1, 1, 1) ** 2))/sigma_h \
        - torch.exp(-.5 * (x_theta ** 2 / sigma_l.view(-1, 1, 1, 1) ** 2 + y_theta ** 2 / sigma_y.view(-1, 1, 1, 1) ** 2))/sigma_l) \
         / (1.0/sigma_h - 1.0/sigma_l)
    
    return gb

# L1 loss of orientation map
class orient(nn.Module):
    def __init__(self, channel_in=1, channel_out=1, stride=1, padding=8):
        super(orient, self).__init__()
        self.criterion = nn.L1Loss()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.stride = stride
        self.padding = padding
        self.filter = DoG_fn

        self.numKernels = 32
        self.kernel_size = 17

        self.filterKernel = []
        for iOrient in range(self.numKernels):
            theta = nn.Parameter(torch.ones(self.channel_out)*(math.pi*iOrient/self.numKernels), requires_grad=False).cuda()
            filterKernel = self.filter(self.kernel_size, self.channel_in, self.channel_out, theta)
            filterKernel = filterKernel.float()
            self.filterKernel.append(filterKernel.squeeze(0).data)
        self.filterKernel = torch.cat(self.filterKernel, dim=0).unsqueeze(1).data
        self.filterKernel.requires_grad = False

    def calOrientation(self, image):

        # filter the image with different orientations
        image = image.cuda()
        resTensor = F.conv2d(image, self.filterKernel, stride=self.stride, padding=self.padding)

        self.resTensor = resTensor
        # argmax the response
        resTensor[resTensor < 0] = 0
        maxResTensor = torch.argmax(resTensor, dim=1).float().data # range from 0 to 31
        confidenceTensor = torch.max(resTensor, dim=1)[0].data

        return maxResTensor, confidenceTensor

    def calc(self, image: torch.Tensor, mask=None):
        if image.min() < -0.1:
            image = (image.clamp(-1, 1) + 1) / 2.0

        fake_image = image * 255
        gray = 0.299 * fake_image[:, 0, :, :] + 0.587 * fake_image[:, 1, :, :] + 0.144 * fake_image[:, 2, :, :]
        gray = torch.unsqueeze(gray, 1)

        resTensor = F.conv2d(gray, self.filterKernel, stride=self.stride, padding=self.padding)
        resTensor[resTensor < 0] = 0
        resTensor = resTensor / 255
        if mask is None:
            return resTensor
        else:
            return resTensor * ((mask > 0.5).float())

    def get_resTensor(self):
        return self.resTensor

#%%
if __name__ == '__main__':
    class args:
        orientation_root = './orient'
        # image_path = './results/0_00018/recface_1200.png'
        # hairmask_path = './results/0_00018/recmask_1200.png'
        image_path = './data/images/00018.jpg'
        hairmask_path = './data/masks/00018.png'
    # mkdir orientation root
    if not os.path.exists(args.orientation_root):
        os.mkdir(args.orientation_root)
        
    # Get structure
    image = Image.open(args.image_path)

    mask = Image.open(args.hairmask_path).convert('RGB')
    mask = transforms.Resize((1024, 1024))(mask)
    mask = transforms.ToTensor()(mask)[0]
    mask = np.array(mask)
    if np.max(mask) > 1:
        mask = (mask > 130) * 1
    trans_image = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image_tensor = trans_image(image)
    image_tensor = torch.unsqueeze(image_tensor, 0)
    cal_orient = orient()
    fake_image = (image_tensor + 1) / 2.0 * 255
    
    gray = 0.299 * fake_image[:, 0, :, :] + 0.587 * fake_image[:, 1, :, :] + 0.144 * fake_image[:, 2, :, :]
    gray = torch.unsqueeze(gray, 1)

    fave_save = Image.fromarray(np.uint8(gray.squeeze()))
 
    orient_tensor, confidence_tensor = cal_orient.calOrientation(gray)
    

    orient_tensor = orient_tensor * math.pi / 31 * 2
    mask_tensor = torch.from_numpy(mask).float().cuda()
    flow_x = torch.cos(orient_tensor) * confidence_tensor
    flow_y = torch.sin(orient_tensor) * confidence_tensor
    # flow_x = torch.from_numpy(cv2.GaussianBlur(flow_x.numpy().squeeze(), (0, 0), 4))
    # flow_y = torch.from_numpy(cv2.GaussianBlur(flow_y.numpy().squeeze(), (0, 0), 4))
    orient_tensor = torch.atan2(flow_y, flow_x) * 0.5
    orient_tensor[orient_tensor < 0] += math.pi
    orient_tensor = orient_tensor.squeeze() * 255. / math.pi * mask_tensor

    orient_numpy = np.array(orient_tensor.cpu())

    select = (confidence_tensor > confidence_tensor.mean()).squeeze().cpu().numpy()
    orient_np = orient_numpy * select
    orient_save = Image.fromarray(np.uint8(orient_np))
    orient_save.save(os.path.join(args.orientation_root, args.image_path.split('/')[-1][:-4]+'.png'))
    # cv2.imwrite(args.orientation_root, orient_tensor.numpy().squeeze() * 255. / math.pi)

# %%
# pd.Series(orient_numpy.reshape(-1)).hist(bins=100, range=[1,255])
# %%
