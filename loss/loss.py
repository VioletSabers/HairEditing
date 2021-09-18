import sys
sys.path.append("./")
from networkx.algorithms.euler import has_eulerian_path
from typing import List
import torch.nn as nn
from loss.custom_loss import prepare_mask, custom_loss
from networks.VGG16 import VGG16_perceptual
from utils.config import cfg
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from networks.orientation import orient


class LPIPSLoss:
    def __init__(self, in_size=1024, out_size=256):
        self.perceptual = nn.DataParallel(VGG16_perceptual()).cuda()
        self.MSE_loss = nn.DataParallel(nn.MSELoss(reduction="mean")).cuda()
        self.upsample = nn.DataParallel(nn.Upsample(scale_factor=out_size / in_size, mode='bilinear'))
        self.insize = in_size
        self.outsize = out_size
    def __call__(self, syn_img, img, mask=None, mul_mask=False):
        assert(syn_img.shape[-1] == self.insize and syn_img.shape[-2] == self.insize)
        assert(img.shape[-1] == self.insize and img.shape[-2] == self.insize)
        if mask is not None:
            assert(mask.shape[-1] == self.insize and mask.shape[-2] == self.insize)
        assert(syn_img.shape == img.shape)
        if self.insize != self.outsize:
            syn_img_p = self.upsample(syn_img)
            img_p = self.upsample(img)
            if mask is not None:
                mask_p = self.upsample(mask)
            else:
                mask_p = None
        else:
            syn_img_p = syn_img
            img_p = img
            mask_p = mask

        if mask_p is None:
            syn0, syn1, syn2, syn3 = self.perceptual(syn_img_p)
            r0, r1, r2, r3 = self.perceptual(img_p)
            per_loss = 0
            per_loss += self.MSE_loss(syn0, r0)
            per_loss += self.MSE_loss(syn1, r1)
            per_loss += self.MSE_loss(syn2, r2)
            per_loss += self.MSE_loss(syn3, r3)

            loss = cfg.I2SLoss.lamb_p * per_loss
            return loss
        else:
            if mul_mask:
                syn0, syn1, syn2, syn3 = self.perceptual(syn_img_p * mask_p) 
            else:
                syn0, syn1, syn2, syn3 = self.perceptual(syn_img_p) 
            r0, r1, r2, r3 = self.perceptual(img_p)

            mask_size = mask_p.shape[-1]
            per_loss = 0
            mask_layer = F.upsample(mask_p, scale_factor=syn0.shape[-1]/mask_size)
            per_loss += self.MSE_loss(syn0 * mask_layer, r0 * mask_layer)
            mask_layer = F.upsample(mask_p, scale_factor=syn1.shape[-1]/mask_size)
            per_loss += self.MSE_loss(syn1 * mask_layer, r1 * mask_layer)
            mask_layer = F.upsample(mask_p, scale_factor=syn2.shape[-1]/mask_size)
            per_loss += self.MSE_loss(syn2 * mask_layer, r2 * mask_layer)
            mask_layer = F.upsample(mask_p, scale_factor=syn3.shape[-1]/mask_size)
            per_loss += self.MSE_loss(syn3 * mask_layer, r3 * mask_layer)
            return cfg.I2SLoss.lamb_p * per_loss

class I2SNoiseLoss:
    def __init__(self):
        self.MSE_loss = nn.MSELoss(reduction="mean")

    def __call__(self, syn_img, *img):
        assert(len(img) == len(cfg.I2SLoss.lamb_noisemse))

        loss = 0
        for i, img_p in enumerate(img):
            loss += cfg.I2SLoss.lamb_noisemse[i] * self.MSE_loss(syn_img, img[i])
        return loss

class StyleLoss(nn.Module):
    def __init__(self, normalize=False, distance="l2"):

        super(StyleLoss, self).__init__()

        self.vgg16 = nn.DataParallel(VGG16_perceptual()).cuda()

        self.normalize = normalize
        self.distance = distance

    def get_features(self, model, x):

        return model(x)

    def mask_features(self, x, mask):

        mask = prepare_mask(x, mask)
        return x * mask

    def gram_matrix(self, x):
        """
        :x is an activation tensor
        """
        N, C, H, W = x.shape
        x = x.view(N * C, H * W)
        G = torch.mm(x, x.t())

        return G.div(N * H * W * C)

    def cal_style(self, model, x, x_hat, mask1=None, mask2=None):
        # Get features from the model for x and x_hat
        with torch.no_grad():
            act_x = self.get_features(model, x)
        for layer in range(0, len(act_x)):
            act_x[layer].detach_()

        act_x_hat = self.get_features(model, x_hat)

        loss = 0.0
        for layer in range(0, len(act_x)):

            # mask features if present
            if mask1 is not None:
                feat_x = self.mask_features(act_x[layer], mask1)
            else:
                feat_x = act_x[layer]
            if mask2 is not None:
                feat_x_hat = self.mask_features(act_x_hat[layer], mask2)
            else:
                feat_x_hat = act_x_hat[layer]

            # compute Gram matrix for x and x_hat
            G_x = self.gram_matrix(feat_x)
            G_x_hat = self.gram_matrix(feat_x_hat)

            # compute layer wise loss and aggregate
            loss += custom_loss(
                G_x, G_x_hat, mask=None, loss_type=self.distance, include_bkgd=True
            )

        loss = loss / len(act_x)

        return loss

    def forward(self, x, x_hat, mask1=None, mask2=None):
        x = x.cuda()
        x_hat = x_hat.cuda()

        # resize images to 256px resolution

        loss = self.cal_style(self.vgg16, x, x_hat, mask1=mask1, mask2=mask2)

        return loss

class SegLoss():
    def __call__(self, mask: torch.Tensor, target_mask: torch.Tensor):
        assert(mask.shape[2:] == target_mask.shape[2:])
        assert(mask.shape[0] == 1 and mask.shape[1] == 3)

        return  -(torch.log(mask[0][0] + 1e-6) * target_mask[0][0]).mean() \
                -(torch.log(mask[0][1] + 1e-6) * target_mask[0][1]).mean() \
                -(torch.log(mask[0][2] + 1e-6) * target_mask[0][2]).mean() * 3 \
                +(torch.log(mask[0][1] + 1e-6) * target_mask[0][3]).mean() \
                +(torch.log(mask[0][2] + 1e-6) * target_mask[0][4]).mean()


class OneclassLoss():
    def __call__(self, mask: torch.Tensor, target_mask: torch.Tensor):
        assert(mask.shape == target_mask.shape)
        loss = -(torch.log(mask) * target_mask).mean()
        return loss

class ShapeLoss():
    def __init__(self):
        self.upsample = nn.DataParallel(nn.Upsample(scale_factor=0.25, mode='bilinear'))
    def __call__(self, mask1, mask2):
        mask1_down1, mask2_down1 = self.upsample(mask1), self.upsample(mask2)
        mask1_down2, mask2_down2 = self.upsample(mask1_down1), self.upsample(mask2_down1)
        mask1_down3, mask2_down3 = self.upsample(mask1_down2), self.upsample(mask2_down2)

        return (mask1_down3 - mask2_down3).norm() + \
             1/16 * (mask1_down2 - mask2_down2).norm()
            

class OrientLoss():
    def __init__(self, image, hair_mask, size=1024):
        assert(image.shape[-1] == size)
        assert(hair_mask.shape[-1] == size)
        self.size = size
        self.orient = orient()
        self.hair_mask = hair_mask

        transfer_image = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if image.min() > -0.01:
            self.image = transfer_image(image)
        else:
            self.image = image

        self.orient_hair, self.confidence_hair = self.orient.calOrientation(self.gengray(self.image))
        self.hair_mask = hair_mask
        
    def gengray(self, image_tensor):
        if image_tensor.min() < -0.01:
            fake_image = (image_tensor + 1) / 2.0 * 255
        else:
            fake_image = image_tensor * 255
        gray = 0.299 * fake_image[:, 0, :, :] + 0.587 * fake_image[:, 1, :, :] + 0.144 * fake_image[:, 2, :, :]
        gray = torch.unsqueeze(gray, 1)
        return gray
    
    def __call__(self, image_syn, hair_mask_syn, sampled_mask = None):
        assert(image_syn.shape[-1] == self.size)
        assert(hair_mask_syn.shape[-1] == self.size)

        loss = 0
        response = F.conv2d(self.gengray(image_syn), self.orient.filterKernel, stride=1, padding=8)
        if sampled_mask is None:
            target = response * hair_mask_syn * self.hair_mask
        else:
            target = response * hair_mask_syn * sampled_mask
        
        for k in range(32):

            loss += ((target - self.confidence_hair) * (self.orient_hair == k) * sampled_mask).norm() / self.size / self.size
            
        return loss
class IMSELoss():
    def __call__(self, img1, img2, mask = None):
        assert(img1.shape == img2.shape)
        if mask is None:
            return (img1 - img2).norm()
        else:
            return ((img1 - img2) * mask).norm() / (img1.shape[-1] ** 2)