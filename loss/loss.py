import sys
sys.path.append("./")
import torch.nn as nn
from loss.custom_loss import prepare_mask, custom_loss
from models.VGG16 import VGG16_perceptual
from configs.global_config import cfg
import torch
import torch.nn.functional as F
from models.orientation import orient
from utils.c_utils import *


class LPIPSLoss:
    def __init__(self, in_size=1024, out_size=256, size=['1_1', '1_2', '3_2', '4_2']):
        self.perceptual = VGG16_perceptual().cuda()
        self.MSE_loss = nn.MSELoss(reduction="mean").cuda()
        self.upsample = nn.Upsample(scale_factor=out_size / in_size, mode='bilinear')
        self.insize = in_size
        self.outsize = out_size
        self.size = size
        
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
            if '1_1' in self.size:
                per_loss += self.MSE_loss(syn0, r0)
            if '1_2' in self.size:
                per_loss += self.MSE_loss(syn1, r1)
            if '3_2' in self.size:
                per_loss += self.MSE_loss(syn2, r2)
            if '4_2' in self.size:
                per_loss += self.MSE_loss(syn3, r3)

            loss = per_loss
            return loss
        else:
            if mul_mask:
                syn0, syn1, syn2, syn3 = self.perceptual(syn_img_p * mask_p)
                r0, r1, r2, r3 = self.perceptual(img_p * mask_p)

                mask_size = mask_p.shape[-1]
                per_loss = 0
                if '1_1' in self.size:
                    per_loss += self.MSE_loss(syn0, r0)
                if '1_2' in self.size:
                    per_loss += self.MSE_loss(syn1, r1)
                if '3_2' in self.size:
                    per_loss += self.MSE_loss(syn2, r2)
                if '4_2' in self.size:
                    per_loss += self.MSE_loss(syn3, r3)
                return per_loss
            else:
                syn0, syn1, syn2, syn3 = self.perceptual(syn_img_p) 
                r0, r1, r2, r3 = self.perceptual(img_p)

                mask_size = mask_p.shape[-1]
                per_loss = 0
                if '1_1' in self.size:
                    mask_layer = F.upsample(mask_p, scale_factor=syn0.shape[-1]/mask_size, mode='nearest')
                    per_loss += self.MSE_loss(syn0 * mask_layer, r0 * mask_layer)
                if '1_2' in self.size:
                    mask_layer = F.upsample(mask_p, scale_factor=syn1.shape[-1]/mask_size, mode='nearest')
                    per_loss += self.MSE_loss(syn1 * mask_layer, r1 * mask_layer)
                if '3_2' in self.size:
                    mask_layer = F.upsample(mask_p, scale_factor=syn2.shape[-1]/mask_size, mode='nearest')
                    per_loss += self.MSE_loss(syn2 * mask_layer, r2 * mask_layer)
                if '4_2' in self.size:
                    mask_layer = F.upsample(mask_p, scale_factor=syn3.shape[-1]/mask_size, mode='nearest')
                    per_loss += self.MSE_loss(syn3 * mask_layer, r3 * mask_layer)
                return per_loss

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
                -(torch.log(mask[0][2] + 1e-6) * target_mask[0][2]).mean() \
                +(torch.log(mask[0][1] + 1e-6) * target_mask[0][3]).mean() * 5 \
                +(torch.log(mask[0][2] + 1e-6) * target_mask[0][4]).mean() * 5


class OneclassLoss():
    def __call__(self, mask: torch.Tensor, target_mask: torch.Tensor):
        assert(mask.shape == target_mask.shape)
        loss = -(torch.log(mask) * target_mask).mean()
        return loss


class OrientLoss():
    def __init__(self, size=1024):
        self.size = size
        self.orient = orient()

    def gengray(self, image_tensor):
        image_tensor = image_tensor - image_tensor.min()
        image_tensor = image_tensor / image_tensor.max()
        fake_image = image_tensor * 255
        gray = 0.299 * fake_image[:, 0, :, :] + 0.587 * fake_image[:, 1, :, :] + 0.144 * fake_image[:, 2, :, :]
        gray = torch.unsqueeze(gray, 1)
        return gray
    
    def calc(self, image_syn):
        assert(image_syn.shape[-1] == self.size)

        response = F.conv2d(self.gengray(image_syn), self.orient.filterKernel, stride=1, padding=8)
        return response


if __name__ == "__main__":
    orient = OrientLoss()
    syn_img = torch.rand(1, 3, 1024, 1024).cuda()
    print(orient.calc(syn_img).shape)
