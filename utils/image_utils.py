from collections import OrderedDict
import torch

import numpy as np
import pickle
import os
from PIL import Image

def getTargetMask(FM, HM):
    FM = FM.squeeze().unsqueeze(0)
    HM = HM.squeeze().unsqueeze(0)
    assert(FM.shape == HM.shape)
    FM = ((FM - HM) > 0.5).float()
    BM = ((FM + HM) < 0.5).float()
    return torch.cat([BM, FM, HM], dim=0)

    

def getImagePaths(raw, mask, *image_path):
    out = OrderedDict()
    for i, img_path in enumerate(image_path):
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


