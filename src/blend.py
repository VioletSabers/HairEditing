
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sys.path.append("./")

import torch
from networks import StyleGAN2
from networks import deeplab_xception_transfer

from utils.config import cfg
from loss.loss import ShapeLoss, StyleLoss, SegLoss, IMSELoss, OrientLoss
from networks.graphonomy_inference import get_mask, get_3class, get_hairclass, get_target_mask

import torch.optim as optim
from torchvision.utils import save_image
from utils import optimizer_utils, image_utils
from datasets.ffhq import process_image
from loss.loss import LPIPSLoss, OneclassLoss
import scipy.ndimage
import numpy as np
from PIL import Image
import math
import torch.nn as nn
from src.faceparsing import Parsing
from hairstyle import Style

def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise

def handle(img):
    img = torch.clamp(img, -1, 1)
    img = (img + 1.0) / 2.0
    return img

def save(image):
    if image.shape[1] == 3:
        image_utils.writeImageToDisk(
            [image], ['temp_image.png'], './results'
        )
    else:
        image_utils.writeMaskToDisk(
            [image], ['temp_mask.png'], './results'
        )


class Blend:
    def __init__(self, image_face, image_hair, mask_face, mask_hair, *image_name):

        # assert
        assert(image_face.shape == (1, 3, cfg.size, cfg.size))
        assert(image_hair.shape == (1, 3, cfg.size, cfg.size))
        assert(mask_face.shape == (1, 3, cfg.size, cfg.size))
        assert(mask_hair.shape == (1, 3, cfg.size, cfg.size))

        insize = cfg.size                       # 1024
        outsize = cfg.image.resize              # 256

        # 构建分割网络
        self.SegNet = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(
            n_classes=20,
            hidden_layers=128,
            source_classes=7,
        )
        state_dict = torch.load(cfg.segment.model_path)
        self.SegNet.load_source_model(state_dict)
        self.SegNet = nn.DataParallel(self.SegNet).cuda()
        self.SegNet.eval()

        # 构建生成网络
        pretrain_path = 'pretrain_model/stylegan2-ffhq-config-f.pt';
        self.G = StyleGAN2.Generator(cfg.align.size, 512, 8)
        self.G.load_state_dict(torch.load(pretrain_path)["g_ema"], strict=False)
        self.G = nn.DataParallel(self.G).cuda()
        self.G.eval()

        # 构建损失函数
        self.oneclassloss = OneclassLoss()
        self.lpipsloss = LPIPSLoss(in_size=1024, out_size=256)
        self.upsample = nn.DataParallel(nn.Upsample(scale_factor=outsize / insize, mode='bilinear')).cuda()
        self.shapeloss = ShapeLoss()
        self.styleloss = nn.DataParallel(StyleLoss()).cuda()
        self.upsample_1024 = nn.Upsample(scale_factor=1024 / 256, mode='bilinear')
        self.upsample = nn.Upsample(scale_factor=256 / 1024, mode='bilinear')
        self.segloss = SegLoss()
        self.mseloss = IMSELoss()
        

        # 类属性处理
        self.mask_face, self.mask_hair = mask_face, mask_hair
        self.image_face, self.image_hair = image_face, image_hair
        self.mask_face_F, self.mask_face_H = mask_face[:,2,:,:].unsqueeze(1), mask_face[:,0,:,:].unsqueeze(1)
        self.mask_hair_F, self.mask_hair_H = mask_hair[:,2,:,:].unsqueeze(1), mask_hair[:,0,:,:].unsqueeze(1)
        
        self.image_face_d, self.image_hair_d = self.upsample(self.image_face), self.upsample(self.image_hair)
        self.mask_face_Fd, self.mask_hair_Hd = self.upsample(self.mask_face_F), self.upsample(self.mask_hair_H)
        self.mask_hair_Fd = self.upsample(self.mask_hair_F)

        self.image_name1 = image_name[0].split('.')[0]
        self.image_name2 = image_name[1].split('.')[0]
        self.task_name = self.image_name1 + '_' + self.image_name2

        self.orientloss = OrientLoss(self.image_hair, self.mask_hair_H)
        # 图像处理
        parsing = Parsing(256)
        self.mask_classifier = parsing.get_Face_Noface(self.upsample(mask_face), self.upsample(mask_hair))

        self.image_hair_style = Style(image_hair, mask_hair)
        self.hair_face_shape = self.image_hair_style.get_faceshape()
        self.hair_face_mid = self.image_hair_style.get_middle()

        self.hair_target_mask = self.image_hair_style.get_importantOrientation()
        self.hair_target_mask = torch.tensor(scipy.ndimage.binary_dilation(self.hair_target_mask.cpu(), iterations=5)).float().cuda()
        self.hair_target_mask = self.hair_target_mask * self.mask_hair_H
        self.hair_target_mask = self.hair_target_mask.data
        self.hair_target_mask.requires_grad = False

        self.hair_target_image = self.hair_target_mask * image_hair
        self.mask_classifier[0,2,:,:] = self.upsample(self.hair_target_mask)

        self.mask_classifier = torch.cat([self.mask_classifier, parsing.get_NoHair(self.mask_hair_Fd, self.mask_hair_Hd)], dim=1)

    def initnoise(self, latent_in_m, noises_m):
        n_mean_latent = 10000
        with torch.no_grad():
            noise_sample = torch.randn(n_mean_latent, 512).cuda()
            latent_out = self.G.module.style(noise_sample)
            
            latent_mean = latent_out.mean(0)
            latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5
        if latent_in_m is None:
            
            latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(self.image_face.shape[0], 1)
            latent_in = latent_in.unsqueeze(1).repeat(1, self.G.module.n_latent, 1)
            latent_in.requires_grad = True
        else:
            latent_in = latent_in_m
            latent_in.requires_grad = True
        
        if noises_m is None:

            noises_single = self.G.module.make_noise()
            noises = []
            for noise in noises_single:
                noises.append(noise.repeat(self.image_face.shape[0], 1, 1, 1).normal_())
            for noise in noises:
                noise.requires_grad = True
        else:
            noises = noises_m
            for noise in noises:
                noise.requires_grad = True
        self.latent_std = latent_std
        return latent_in, noises

    def __call__(self, latent_in_m=None, noises_m=None):

        print('start recface')
        latent_in, noises = self.initnoise(latent_in_m, noises_m)
        opt = optim.Adam([latent_in] + noises, lr=cfg.align.lr, betas=(0.9, 0.999), eps=1e-8)

        for e in range(cfg.recface.w_epochs):
            print(f"\rEpoch {e}", end="")
            opt.zero_grad()
            noise_strength = self.latent_std * cfg.align.noise * max(0, 1 - e / cfg.align.step / cfg.align.noise_ramp) ** 2
            latent_n = latent_noise(latent_in, noise_strength.item())
            syn_img, _ = self.G([latent_n], input_is_latent=True, noise=noises)
            self.syn_img = syn_img.data
            syn_img_d = self.upsample(syn_img)
            
            lpipsloss = self.lpipsloss(self.image_face, handle(syn_img), self.mask_face_F)
            
            if e < 50 or e % 50 == 0:
                
                predictions, HM, FM = get_mask(self.SegNet, syn_img_d)
                HM, FM = HM.float(), FM.float()
                # hair_shapeloss = self.shapeloss(get_hairclass(predictions).unsqueeze(0), self.mask_hair_Hd);
                classifier_loss = self.segloss(get_3class(predictions), self.mask_classifier)

                # lpipsloss_face = self.lpipsloss(self.image_face, handle(syn_img), self.upsample_1024(FM.unsqueeze(0)))
            else:
                classifier_loss = 0
                # lpipsloss_face = 0
            # styleloss = self.styleloss(self.image_hair_d, handle(syn_img_d), self.mask_hair_d, HM.unsqueeze(0))
            lpipsloss_hair = self.lpipsloss(self.image_hair, handle(syn_img), self.hair_target_mask, True)
            
            # mseloss_hair = self.mseloss(handle(syn_img), self.hair_target_image, self.hair_target_mask)
            # print(mseloss_hair)
                
            if e < 500:
                loss = lpipsloss + (cfg.recface.lamb_classifierloss_1) * classifier_loss
            else:
                # orientloss = self.orientloss(handle(syn_img), self.upsample_1024(HM.unsqueeze(0)), self.hair_target_mask)
                loss = lpipsloss +\
                (cfg.recface.lamb_classifierloss_2) * classifier_loss + \
                (cfg.recface.lamb_lpipsloss) * lpipsloss_hair

            loss.backward()
            opt.step()

            if (e + 1) % 500 == 0:
                print("\riter{}: loss -- {}".format(e + 1, loss.item()))
                save_image(handle(syn_img), "results/" + self.task_name + '/' + "recface_{}.png".format(e + 1))
      
                mask_final = get_3class(predictions)
                image_utils.writeImageToDisk(
                    [mask_final.clone()], [f'recmask_{str(e+1)}.png'], './results/' + self.task_name
                )

        return latent_in, noises
    def get_layerout(self, size=32):
        return self.G.module.get_layerout(size)
    
    def get_synimg(self):
        return self.syn_img
    
    def get_synmask(self):
        return get_3class(self.predictions)

if __name__ == '__main__':
    raw = "data/images"
    mask = "data/masks"
    background = "data/backgrounds"
    softmask = "data/softmasks"
    image1 = "0.jpg"
    image2 = "67172.jpg"
    image_files = image_utils.getImagePaths(raw, mask, background, image1, image2)
    I_1, M_1, HM_1, H_1, FM_1, F_1 = process_image(
        image_files['I_1_path'], image_files['M_1_path'], size=1024, normalize=1)

    I_2, M_2, HM_2, H_2, FM_2, F_2 = process_image(
        image_files['I_2_path'], image_files['M_2_path'], size=1024, normalize=1)


    I_1, M_1, HM_1, H_1, FM_1, F_1 = optimizer_utils.make_cuda(
        [I_1, M_1, HM_1, H_1, FM_1, F_1])
    I_2, M_2, HM_2, H_2, FM_2, F_2 = optimizer_utils.make_cuda(
        [I_2, M_2, HM_2, H_2, FM_2, F_2])

    I_1, M_1, HM_1, H_1, FM_1, F_1= image_utils.addBatchDim(
        [I_1, M_1, HM_1, H_1, FM_1, F_1])
    I_2, M_2, HM_2, H_2, FM_2, F_2 = image_utils.addBatchDim(
        [I_2, M_2, HM_2, H_2, FM_2, F_2])
    
    I_1, I_2 = handle(I_1), handle(I_2)


    

    task_name = image1.split('.')[0] + '_' + image2.split('.')[0]
    # if not os.path.exists('./results/' + image1.split('.')[0]):
    #     os.mkdir('./results/' + image1.split('.')[0])
    # if not os.path.exists('./results/' + image2.split('.')[0]):
    #     os.mkdir('./results/' + image2.split('.')[0])
    if not os.path.exists('./results/' + task_name):
        os.mkdir('./results/' + task_name)

    blend_name = image1.split('.')[0] + '_' + image2.split('.')[0]
    
    image_utils.writeImageToDisk(
            [I_1 * FM_1, I_2 * HM_2], ['Face.png', 'Hair.png'], './results/' + blend_name
        )

    recface = Blend(I_1, I_2, M_1, M_2, image1, image2)
    
    latent_in_1, noises_1 = recface()
