
import sys
import os
from typing import Tuple


import torch

from utils_c.config import cfg

from loss.loss import LPIPSLoss
from networks.graphonomy_inference import get_mask, get_3class
from src.Base import *
import torch.optim as optim
from torchvision.utils import save_image
from utils_c import image_utils
# from src.faceparsing import Parsing
import copy


import torch.nn as nn
from src.faceparsing import Parsing
import genmask

from utils_c import image_utils as ImgU

class Blend(BaseClass):
    def __init__(self, face, hair, target, *image_name):
        super(Blend, self).__init__()
        # assert
        assert(len(face) == 3)
        assert(len(hair) == 3)
        image_face, mask_face, F_face = face[0], face[1], face[2]
        image_hair, mask_hair, F_hair = hair[0], hair[1], hair[2]
        image_tar, mask_tar, F_tar = target[0], target[1], target[2]
        mask_other = 1 - mask_face.sum(dim=1)
        image_face: torch.Tensor
        mask_face: torch.Tensor
        F_face: torch.Tensor

        image_hair: torch.Tensor
        mask_hair: torch.Tensor
        F_hair: torch.Tensor

        image_tar: torch.Tensor
        mask_tar: torch.Tensor
        F_tar: torch.Tensor

        assert(image_face.shape == (1, 3, cfg.size, cfg.size))
        assert(image_hair.shape == (1, 3, cfg.size, cfg.size))
        assert(mask_face.shape == (1, 3, cfg.size, cfg.size))
        assert(mask_hair.shape == (1, 3, cfg.size, cfg.size))
        assert(image_tar.shape == (1, 3, cfg.size, cfg.size))
        assert(mask_tar.shape == (1, 3, cfg.size, cfg.size))
    

        self.lpipsloss = LPIPSLoss(in_size=1024, out_size=256)

        # 类属性处理
        self.mask_face_1024, self.mask_hair_1024 = mask_face, mask_hair
        self.image_face_1024, self.image_hair_1024 = image_face, image_hair
        self.mask_face_F1024, self.mask_face_H1024 = mask_face[:,1,:,:].unsqueeze(1), mask_face[:,2,:,:].unsqueeze(1)
        self.mask_hair_F1024, self.mask_hair_H1024 = mask_hair[:,1,:,:].unsqueeze(1), mask_hair[:,2,:,:].unsqueeze(1)
        self.mask_other_1024 = mask_other.unsqueeze(1)

        self.image_name1 = image_name[0].split('.')[0]
        self.image_name2 = image_name[1].split('.')[0]
        self.task_name = self.image_name1 + '_' + self.image_name2

        self.F_face = F_face
        self.F_hair = F_hair
        self.F_tar = F_tar

        self.mask_face_Fmid = self.resample(self.mask_face_F1024, 1024, cfg.mid_size)
        self.mask_hair_Hmid = self.resample(self.mask_hair_H1024, 1024, cfg.mid_size)

        
    def latent_noise(self, latent, strength):
        noise = torch.randn_like(latent) * strength
        return latent + noise
    
    def save_log(self, e, loss, syn_img, predictions=None):
        with torch.no_grad():
            print("\riter{}: loss -- {}".format(e, loss.item()))
            save_image(ImgU.handle(syn_img), "results/" + self.task_name + '/' + "recface_{}.png".format(e))
            if predictions is not None:
                mask_final = get_3class(predictions)
                image_utils.writeImageToDisk(
                    [mask_final.clone()], [f'recmask_{str(e)}.png'], './results/' + self.task_name
                )
    
    def __call__(self, latent_in_m=None, noises_m=None):
        
        print('start blend stage 1')
        latent_in, noises, latent_std = self.initnoise(latent_in_m, noises_m)

        opt = optim.Adam([latent_in] + noises, lr=cfg.stage1.lr, betas=(0.9, 0.999), eps=1e-8)
        for e in range(0, cfg.stage1.epochs):
            print(f"\rEpoch {e}", end="")
            opt.zero_grad()
            noise_strength = latent_std * cfg.styleGAN.noise * max(0, 1 - e / cfg.styleGAN.step / cfg.styleGAN.noise_ramp) ** 2
            latent_n = self.latent_noise(latent_in, noise_strength.item())
            syn_img, _ = G([latent_n], input_is_latent=True, noise=noises)
            
            layer_outmid, skip = G.get_layerout(cfg.mid_size)

            loss = mseloss(layer_outmid, self.F_tar[cfg.mid_size][0])
            loss.backward()
            opt.step()
            if (e + 1) % cfg.print_epoch == 0:
                self.save_log(e + 1, loss, syn_img, None)

        syn_img_d = self.resample(syn_img, 1024, 256)
        predictions, HM, FM, BM = get_mask(SegNet, syn_img_d)
        HM, FM = HM.unsqueeze(0).float(), FM.unsqueeze(0).float()

        print('start blend stage 2')
        opt = optim.Adam([latent_in] + noises, lr=cfg.stage2.lr, betas=(0.9, 0.999), eps=1e-8)
        latent_in_init = copy.deepcopy(latent_in.detach().data)
        latent_in_init.requires_grad = False

        for e in range(cfg.stage1.epochs, cfg.stage1.epochs + cfg.stage2.epochs):
            print(f"\rEpoch {e}", end="")
            opt.zero_grad()
            noise_strength = latent_std * cfg.styleGAN.noise * max(0, 1 - e / cfg.styleGAN.step / cfg.styleGAN.noise_ramp) ** 2
            latent_n = self.latent_noise(latent_in, noise_strength.item())
            MASK_H = self.resample(HM, 256, 1024) * self.mask_hair_H1024
            MASK_F = self.resample(FM, 256, 1024) * self.mask_hair_F1024 + self.mask_other_1024 * (1 - MASK_H)
            syn_img, _ = G([latent_n], input_is_latent=True, noise=noises)
            hair_lpipsloss = self.lpipsloss(ImgU.handle(syn_img), self.image_hair_1024, MASK_H)
            face_lpipsloss = self.lpipsloss(ImgU.handle(syn_img), self.image_face_1024, MASK_F)
            loss = hair_lpipsloss + face_lpipsloss +\
                cfg.stage2.lamb_mseloss_1024 * mseloss(ImgU.handle(syn_img) * MASK_H, self.image_hair_1024 * MASK_H)
            loss.backward()
            opt.step()

            latent_in.requires_grad = False
            latent_in[:,9:,:] = latent_in_init[:,9:,:].data
            latent_in.requires_grad = True
            if (e + 1) % cfg.print_epoch == 0: 
                self.save_log(e+1, loss, syn_img, predictions)
        
        print('start blend stage 3')
        opt = optim.Adam([latent_in] + noises, lr=cfg.stage3.lr, betas=(0.9, 0.999), eps=1e-8)
        latent_in_init = copy.deepcopy(latent_in.detach().data)
        latent_in_init.requires_grad = False
        for e in range(cfg.stage1.epochs + cfg.stage2.epochs, cfg.stage1.epochs + cfg.stage2.epochs + cfg.stage3.epochs):
            print(f"\rEpoch {e}", end="")
            opt.zero_grad()
            noise_strength = latent_std * cfg.styleGAN.noise * max(0, 1 - e / cfg.styleGAN.step / cfg.styleGAN.noise_ramp) ** 2
            latent_n = self.latent_noise(latent_in, noise_strength.item())
            MASK_H = self.resample(HM, 256, 1024) * self.mask_hair_H1024
            MASK_F = self.resample(FM, 256, 1024) * self.mask_hair_F1024 + self.mask_other_1024 * (1 - MASK_H)
            syn_img, _ = G([latent_n], input_is_latent=True, noise=noises)
            hair_lpipsloss = self.lpipsloss(ImgU.handle(syn_img), self.image_hair_1024, MASK_H)
            face_lpipsloss = self.lpipsloss(ImgU.handle(syn_img), self.image_face_1024, MASK_F)
            loss = hair_lpipsloss + face_lpipsloss +\
                cfg.stage3.lamb_mseloss_1024 * mseloss(ImgU.handle(syn_img) * MASK_H, self.image_hair_1024 * MASK_H)
            loss.backward()
            opt.step()

            latent_in.requires_grad = False
            latent_in[:, :9].data = latent_in_init[:, :9].data
            latent_in.requires_grad = True
            if (e + 1) % cfg.print_epoch == 0: 
                self.save_log(e+1, loss, syn_img, predictions)
        F, _ = G.get_layerout(cfg.mid_size)
        
        return F, latent_in, ImgU.handle(syn_img)