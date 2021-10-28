from PIL import Image
from torchvision import transforms
from math import log10
import torch


def get_device(use_cuda=True):
    if use_cuda:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cpu")


def image_reader(img_path):
    with open(img_path,"rb") as f:
        image=Image.open(f)
        image=image.convert("RGB")
    transform = transforms.Compose([
        transforms.CenterCrop((1024, 1024)),
        transforms.ToTensor()
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image


def loss_function(syn_img, img, img_p, MSE_loss, upsample, perceptual):
    syn_img_p = upsample(syn_img)
    syn0, syn1, syn2, syn3 = perceptual(syn_img_p)
    r0, r1, r2, r3 = perceptual(img_p)
    mse = MSE_loss(syn_img, img)

    per_loss = 0
    per_loss += MSE_loss(syn0,r0)
    per_loss += MSE_loss(syn1,r1)
    per_loss += MSE_loss(syn2,r2)
    per_loss += MSE_loss(syn3,r3)

    return mse, per_loss

def Mst_loss(syn_image, image, upsample, lamb_s, MSE_loss, style, M_s = None):
    '''
        For style loss
    '''
    if M_s is not None:
        syn_img_p = upsample(M_s * syn_image)
        img_p = upsample(M_s * image)
    else:
        syn_img_p = upsample(syn_image)
        img_p = upsample(image)

    syn_style = style(syn_img_p)
    img_style = style(img_p)

    loss = lamb_s * MSE_loss(syn_style, img_style)
    return loss


def PSNR(mse, flag = 0):
    if flag == 0:
        psnr = 10 * log10(1 / mse.item())
    return psnr