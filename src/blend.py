
import sys
import os
from typing import Tuple


import torch

from utils.config import cfg

from loss.loss import LPIPSLoss
from networks.graphonomy_inference import get_mask, get_3class
from src.Base import *
import torch.optim as optim
from torchvision.utils import save_image
from utils import image_utils
from src.faceparsing import Parsing


import torch.nn as nn
from src.faceparsing import Parsing
import genmask

from utils import image_utils as ImgU

class Blend(BaseClass):
    def __init__(self, face, hair, target_mask, apperance, *image_name):
        super(Blend, self).__init__()
        # assert
        assert(len(face) == 3)
        assert(len(hair) == 3)
        image_face, mask_face, F_init = face[0], face[1], face[2]
        image_hair, mask_hair, H_init = hair[0], hair[1], hair[2]
        image_face: torch.Tensor
        mask_face: torch.Tensor
        F_init: torch.Tensor

        image_hair: torch.Tensor
        mask_hair: torch.Tensor
        H_init: torch.Tensor

        target_mask: torch.Tensor

        assert(image_face.shape == (1, 3, cfg.size, cfg.size))
        assert(image_hair.shape == (1, 3, cfg.size, cfg.size))
        assert(mask_face.shape == (1, 3, cfg.size, cfg.size))
        assert(mask_hair.shape == (1, 3, cfg.size, cfg.size))
        assert((target_mask is None) or (target_mask.shape == (1, 3, cfg.size, cfg.size)))

        mid_size = cfg.mid_size              # 32
        self.mid_size = mid_size
        self.lpipsloss = LPIPSLoss(in_size=1024, out_size=256)

        # 类属性处理
        self.mask_face_1024, self.mask_hair_1024 = mask_face, mask_hair
        self.image_face_1024, self.image_hair_1024 = image_face, image_hair
        self.mask_face_F1024, self.mask_face_H1024 = ImgU.erosion(mask_face[:,2,:,:].unsqueeze(1), iteration=8), ImgU.erosion(mask_face[:,0,:,:].unsqueeze(1), iteration=8)
        self.mask_hair_F1024, self.mask_hair_H1024 = ImgU.erosion(mask_hair[:,2,:,:].unsqueeze(1), iteration=8), ImgU.erosion(mask_hair[:,0,:,:].unsqueeze(1), iteration=8)
        
        self.image_face_256, self.image_hair_256 = self.resample(self.image_face_1024, 1024, 256), self.resample(self.image_hair_1024, 1024, 256)
        self.mask_face_F256, self.mask_hair_H256 = self.resample(self.mask_face_F1024, 1024, 256), self.resample(self.mask_hair_H1024, 1024, 256)
        self.mask_hair_F256, self.mask_face_H256 = self.resample(self.mask_hair_F1024, 1024, 256), self.resample(self.mask_face_H1024, 1024, 256)
        if target_mask is not None:
            self.mask_target_H1024 = target_mask[:,0,:,:].unsqueeze(0)
            self.mask_target_F1024 = target_mask[:,2,:,:].unsqueeze(0)
        else:
            self.mask_target_H1024 = self.mask_hair_H1024
            self.mask_target_F1024 = self.mask_face_F1024
        self.mask_target_H256 = self.resample(self.mask_target_H1024, 1024, 256)
        self.mask_target_F256 = self.resample(self.mask_target_F1024, 1024, 256)
        self.image_name1 = image_name[0].split('.')[0]
        self.image_name2 = image_name[1].split('.')[0]
        self.task_name = self.image_name1 + '_' + self.image_name2
        self.F_init_face = F_init
        self.F_init_hair = H_init

        # 图像处理
        if target_mask is not None:
            bg = ((1 - self.mask_target_H256 - self.mask_target_F256) > 0.5).float()
            self.mask_segment = torch.cat([bg, self.mask_target_F256, self.mask_target_H256], dim=1)
        else:
            parsing = Parsing(cfg.resize)
            self.mask_segment = parsing.get_Face_Noface(self.resample(mask_face, 1024, 256), self.resample(mask_hair, 1024, 256))

            self.mask_segment[0,1,:,:] = self.mask_target_F256 * (1-self.mask_target_H256)
            self.mask_segment[0,2,:,:] = self.mask_target_H256

            self.mask_segment = torch.cat([self.mask_segment, parsing.get_NoHair(self.mask_target_F256, self.mask_target_H256)], dim=1)


        self.mask_hair_Hmid = self.resample(self.mask_hair_H1024, 1024, mid_size)
        self.mask_face_Fmid = self.resample(self.mask_face_F1024, 1024, mid_size)

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
        
        print('start stage 1')
        latent_in, noises, latent_std = self.initnoise(latent_in_m, noises_m)

        opt = optim.Adam([latent_in] + noises, lr=cfg.stage1.lr, betas=(0.9, 0.999), eps=1e-8)
        for e in range(0, cfg.stage1.epochs):
            print(f"\rEpoch {e}", end="")
            opt.zero_grad()
            noise_strength = latent_std * cfg.styleGAN.noise * max(0, 1 - e / cfg.styleGAN.step / cfg.styleGAN.noise_ramp) ** 2
            latent_n = self.latent_noise(latent_in, noise_strength.item())
            syn_img, _ = G([latent_n], input_is_latent=True, noise=noises)
            syn_img_d = self.resample(syn_img, 1024, 256)
            layer_out32, _ = G.get_layerout(32)

            if e < 150:
                predictions, HM, FM = get_mask(SegNet, syn_img_d)
                HM, FM = HM.unsqueeze(0).float(), FM.unsqueeze(0).float()
                
                segment_loss = segloss(get_3class(predictions), self.mask_segment)
                HM_1024 = self.resample(HM, 256, 1024)
                loss = cfg.stage1.lamb_segmentloss * segment_loss 
            else:
                hair_styleloss = styleloss(self.image_hair_1024, ImgU.handle(syn_img), self.mask_hair_H1024, HM_1024)
                hair_mseloss32 = mseloss(layer_out32 * self.mask_hair_Hmid, self.F_init_hair[self.mid_size][0] * self.mask_hair_Hmid)
                loss = cfg.stage1.lamb_styleloss * hair_styleloss + cfg.stage1.lamb_mseloss_32 * hair_mseloss32
            loss.backward()
            opt.step()
            if (e + 1) % 100 == 0:
                self.save_log(e + 1, loss, syn_img, predictions)
        if cfg.stage1.epochs != 0:
            F = layer_out32.data
            F.requires_grad = True
            S = latent_in
            S.requires_grad = True

            opt = optim.Adam([F, S] + noises, lr=cfg.stage1.lr, betas=(0.9, 0.999), eps=1e-8)
      
        if cfg.stage1.epochs != 0:
            torch.save(F, './results/' + self.task_name + '/blend_stage1_F.pth')
            torch.save(S, './results/' + self.task_name + '/blend_stage1_S.pth')
        else:
            with torch.no_grad():
                F = torch.load('./results/' + self.task_name + '/blend_stage1_F.pth')
                S = torch.load('./results/' + self.task_name + '/blend_stage1_S.pth')
                syn_img = G.mid_start(F, S, noises, None)
                syn_img_d = self.resample(syn_img, 1024, 256)
                predictions, HM, FM = get_mask(SegNet, syn_img_d)
                HM, FM = HM.unsqueeze(0).float(), FM.unsqueeze(0).float()
                
                HM_1024 = self.resample(HM, 256, 1024)


        MASK_F_32 = self.resample(FM, 256, self.mid_size) * self.mask_face_Fmid
        F = (F * (1 - MASK_F_32) + self.F_init_face[self.mid_size][0] * MASK_F_32).data
        S = S.data
        F.requires_grad = True
        S.requires_grad = True
        opt = optim.Adam([F, S] + noises, lr=cfg.stage1.lr, betas=(0.9, 0.999), eps=1e-8)
        print('start stage 2')
        for e in range(cfg.stage1.epochs, cfg.stage1.epochs + cfg.stage2.epochs):
            print(f"\rEpoch {e}", end="")
            opt.zero_grad()
            noise_strength = latent_std * cfg.styleGAN.noise * max(0, 1 - e / cfg.styleGAN.step / cfg.styleGAN.noise_ramp) ** 2
            latent_n = self.latent_noise(latent_in, noise_strength.item())
            MASK_F = self.resample(FM, 256, 1024) * self.mask_face_F1024
            MASK_H = self.resample(HM, 256, 1024) * self.mask_hair_H1024
            syn_img = G.mid_start(F, S, noises, None)
            face_lpipsloss = self.lpipsloss(ImgU.handle(syn_img), self.image_face_1024, MASK_F)
            hair_lpipsloss = self.lpipsloss(ImgU.handle(syn_img), self.image_hair_1024, MASK_H)
            loss = face_lpipsloss + hair_lpipsloss + \
                cfg.stage2.lamb_mseloss_1024 * mseloss(ImgU.handle(syn_img) * MASK_F, self.image_face_1024 * MASK_F)
            loss.backward()
            opt.step()
            if (e + 1) % 100 == 0:
                self.save_log(e+1, loss, syn_img, predictions)
        return ImgU.handle(syn_img)

    def blend_final(self, F_final, S_final):
        print('start stage 3')
        latent_in, noises, latent_std = self.initnoise(None, None)
        # latent_in = S_final.data
        # latent_in.requires_grad = True
        opt = optim.Adam([latent_in] + noises, lr=cfg.stage3.lr, betas=(0.9, 0.999), eps=1e-8)
        for e in range(cfg.stage1.epochs + cfg.stage2.epochs, cfg.stage1.epochs + cfg.stage2.epochs +cfg.stage3.epochs):
            print(f"\rEpoch {e}", end="")
            opt.zero_grad()
            noise_strength = latent_std * cfg.styleGAN.noise * max(0, 1 - e / cfg.styleGAN.step / cfg.styleGAN.noise_ramp) ** 2
            latent_n = self.latent_noise(latent_in, noise_strength.item())
            syn_img, _ = G([latent_n], input_is_latent=True, noise=noises)
            layer_out32, _ = G.get_layerout(32)

            MASK_H_32 = self.resample(self.mask_target_H1024, 1024, self.mid_size)
            hair_mseloss32 = mseloss(layer_out32 * MASK_H_32, F_final[self.mid_size][0] * MASK_H_32)
            loss = hair_mseloss32
            loss.backward()
            opt.step()
            if (e + 1) % 100 == 0:
                self.save_log(e+1, loss, syn_img, None)
        
        F, _ = G.get_layerout(32)
        S = latent_in
        F = F.data
        F.requires_grad = True
        S = S.data
        S.requires_grad = True
        syn_img_S = syn_img.data

        opt = optim.Adam([F, S] + noises, lr=cfg.stage3.lr, betas=(0.9, 0.999), eps=1e-8)
        for e in range(cfg.stage1.epochs + cfg.stage2.epochs + cfg.stage3.epochs, cfg.stage1.epochs + cfg.stage2.epochs + cfg.stage3.epochs * 2):
            print(f"\rEpoch {e}", end="")
            opt.zero_grad()
            noise_strength = latent_std * cfg.styleGAN.noise * max(0, 1 - e / cfg.styleGAN.step / cfg.styleGAN.noise_ramp) ** 2

            syn_img = G.mid_start(F, S, noises, None)
            MASK_H = self.mask_target_H1024
            MASK_F = self.mask_target_F1024 * (1 - MASK_H)
            hair_mseloss = self.lpipsloss(ImgU.handle(syn_img) * MASK_H, ImgU.handle(syn_img_S) * MASK_H)
            face_mseloss = mseloss(ImgU.handle(syn_img) * MASK_F, self.image_face_1024 * MASK_F)
            loss = cfg.stage3.lamb_mseloss_1024 * hair_mseloss + cfg.stage3.lamb_mseloss_1024 * face_mseloss
            if (e + 1) % 100 == 0:
                self.save_log(e+1, loss, syn_img, None)
            loss.backward()
            opt.step()
    def get_layerout(self, size=32):
        return G.get_layerout(size)
    
    def get_synimg(self):
        return self.syn_img
    
    def get_synmask(self):
        return get_3class(self.predictions)