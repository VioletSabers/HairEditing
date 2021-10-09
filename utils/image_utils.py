from collections import OrderedDict
import torch

import numpy as np
import pickle
import os
from PIL import Image
import scipy.ndimage

def handle(img):
    img = torch.clamp(img, -1, 1)
    img = (img + 1.0) / 2.0
    return img

def dilation(mask: torch.Tensor, iteration=10):
    mask = mask.squeeze()
    mask = scipy.ndimage.binary_dilation(
        mask.squeeze().cpu().numpy(), iterations=iteration
    )
    mask = torch.from_numpy(mask).float().cuda()
    return mask.unsqueeze(0).unsqueeze(0)

def erosion(mask: torch.Tensor, iteration=10):
    mask = mask.squeeze()
    mask = scipy.ndimage.binary_erosion(
        mask.squeeze().cpu().numpy(), iterations=iteration
    )
    mask = torch.from_numpy(mask).float().cuda()
    return mask.unsqueeze(0).unsqueeze(0)

def getTargetMask(FM, HM):
    FM = FM.squeeze().unsqueeze(0)
    HM = HM.squeeze().unsqueeze(0)
    assert(FM.shape == HM.shape)
    FM = ((FM - HM) > 0.5).float()
    BM = ((FM + HM) < 0.5).float()
    return torch.cat([BM, FM, HM], dim=0)

def getImagePaths(raw_face, raw_hair, mask, *image_path):
    assert(len(image_path) == 2)
    out = OrderedDict()
    for i, img_path in enumerate(image_path):
        if i == 0:
            raw = raw_face
        else:
            raw = raw_hair
        I_path = os.path.join(raw, img_path)
        M_path = os.path.join(mask, img_path.split('.')[0] + '.png')
        out['I_' + str(i+1) + '_path'] = I_path
        out['M_' + str(i+1) + '_path'] = M_path

    return out


def addBatchDim(listOfVariables):
    modifiedVariables = []

    for var in listOfVariables:
        var = torch.unsqueeze(var, axis=0)
        modifiedVariables.append(var)

    return modifiedVariables


def writeMaskToDisk(li, names, dest):
    for idx, var in enumerate(li):
        var = makeMask(var)
        var = var[0, :, :, 0]
        var = Image.fromarray(var)
        var.save(os.path.join(dest, names[idx]))


def makeMask(tensor):
    return (
        tensor.detach()
        .clamp_(min=0, max=1)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )


def writeImageToDisk(li, names, dest):
    for idx, var in enumerate(li):
        var = makeImage(var)
        var = var[0]
        var = Image.fromarray(var)
        var.save(os.path.join(dest, names[idx]))


def makeImage(tensor):
    return (
        tensor.detach()
        .clamp_(min=0, max=1)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )

def writePickleToDisk(li, names, dest):
    for idx, var in enumerate(li):
        with open(os.path.join(dest, names[idx]), "wb") as handle:
            pickle.dump(
                li[idx].detach().cpu().numpy(),
                handle
            )

def getImageMiddle(img: torch.Tensor):
    img = img.squeeze()
    assert(img.shape[-1] == img.shape[-2])

    index = torch.tensor(np.arange(img.shape[-1]))
    corr = torch.meshgrid(index, index)

    x = corr[0][img==1]
    y = corr[1][img==1]

    return int(x.float().mean().item()), int(y.float().mean().item())

def save(image_m):
    image = image_m.data
    while len(image.shape) < 4:
        image = image.unsqueeze(0)
    if image.shape[1] == 3:
        writeImageToDisk(
            [image], ['temp_image.png'], './results'
        )
    else:
        writeMaskToDisk(
            [image], ['temp_mask.png'], './results'
        )

def RGBtoHSV(C: torch.Tensor):
    R = C[:,0,:,:].unsqueeze(1)
    G = C[:,1,:,:].unsqueeze(1)
    B = C[:,2,:,:].unsqueeze(1)
    max, argmax = torch.max(C, dim=1)

    min, argmin = torch.min(C, dim=1)

    max = max.unsqueeze(1)
    argmax = argmax.unsqueeze(1)
    min = min.unsqueeze(1)
    V, _ = torch.max(C, dim=1)
    V = V.unsqueeze(0)
    S = (max-min)/max
    H = (G-B)/(max-min)* 60 * (argmax == 0).float()
    H = H + (120 + (B-R)/(max-min) * 60) * (argmax == 1).float()
    H = H + (240 + (R-G)/(max-min) * 60) * (argmax == 2).float()
    
    H = (H + 360) % 360
    return torch.cat([H, S, V], dim=1)

def HSVtoRGB(C: torch.Tensor):
    H = C[:,0,:,:].unsqueeze(1)
    S = C[:,1,:,:].unsqueeze(1)
    V = C[:,2,:,:].unsqueeze(1)
    zeros = (S < 1e-4).float()
    angel = (H / 60).int()

    f = H / 60 - angel
    a = V * (1 - S)
    b = V * (1 - S * f)
    c = V * (1 - S * (1 - f))
    
    R = ((angel == 0).float() * V + \
        (angel == 1).float() * b + \
        (angel == 2).float() * a + \
        (angel == 3).float() * a + \
        (angel == 4).float() * c + \
        (angel == 5).float() * V) * (1 - zeros) + V * zeros
    
    G = ((angel == 0).float() * c + \
        (angel == 1).float() * V + \
        (angel == 2).float() * V + \
        (angel == 3).float() * b + \
        (angel == 4).float() * a + \
        (angel == 5).float() * a) * (1 - zeros) + V * zeros
    
    B = ((angel == 0).float() * a + \
        (angel == 1).float() * a + \
        (angel == 2).float() * c + \
        (angel == 3).float() * V + \
        (angel == 4).float() * V + \
        (angel == 5).float() * b) * (1 - zeros) + V * zeros
    
    return torch.cat([R, G, B], dim=1)