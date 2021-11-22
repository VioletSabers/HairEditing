import torch
from PIL import Image

import scipy.ndimage
import torch
from torchvision.transforms import transforms
import os

def get_image_basename(x: str): #通过图像路径获得图像名称

    file_name = x.split('/')[-1]
    base_name = file_name.split('.')[0]
    return base_name

def get_homogeneous_image(input: torch.Tensor): #将图像或者feature map修改成N*C*H*W的形式
    assert(2 <= len(input.shape) <= 4)
    while (len(input.shape) < 4):
        input = input.unsqueeze(0)
    return input

def load_image(img_path, size=1024, normalize=True):
    img = Image.open(img_path).convert("RGB")
    img = transforms.Resize((size, size))(img)
    img = transforms.ToTensor()(img)

    if normalize :
        img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)
    return get_homogeneous_image(img)

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

def writeMaskToDisk(li, names, dest):
    for idx, var in enumerate(li):
        var = makeMask(var)
        var = var[0, :, :, 0]
        var = Image.fromarray(var)
        var.save(os.path.join(dest, names[idx]))

def save_img(image_m, path=None, file=None):
    image = image_m.data
    while len(image.shape) < 4:
        image = image.unsqueeze(0)
    minn = image_m.min()
    maxn = image_m.max()
    if minn < 0:
        image = image / (maxn - minn)
        image = image - image.min()
    
    if path is None:
        path = "./results"

    os.makedirs(path, exist_ok=True)

    if file is None:
        if image.shape[1] == 3:
            writeImageToDisk(
                [image], ['temp_image.png'], path
            )
        else:
            writeMaskToDisk(
                [image], ['temp_mask.png'], path
            )
    else:
        if image.shape[1] == 3:
            writeImageToDisk(
                [image], [file], path
            )
        else:
            writeMaskToDisk(
                [image], [file], path
            )

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

def change_img_TO01(image):
    image = torch.clamp(image, -1, 1)
    image = (image + 1.0) / 2.0
    return image