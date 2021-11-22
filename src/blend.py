import torch
import  torch.optim as optim
import sys
import copy

from torch.serialization import save

sys.path.append("./")

from models.segment.graphonomy_inference import get_mask, get_3class
from Base import *
from utils.c_utils import *

from configs.global_config import cfg

class Blend():
    def __init__(self, image_face_1024, image_hair_1024, mask_face_1024, mask_hair_1024, task_name):
        super(Blend, self).__init__()
        self.image_face_1024, self.mask_face_F1024 = image_face_1024, mask_face_1024[:, 1, :].unsqueeze(1)
        self.image_hair_1024, self.mask_hair_H1024 = image_hair_1024, mask_hair_1024[:, 2, :].unsqueeze(1)
        
        self.mask_BODY_1024 = 1 - mask_face_1024.sum(dim=1).unsqueeze(1)
        
        self.lpipsloss = LPIPSLoss(in_size=1024, out_size=256)
        self.task_name = task_name
        self.Upsample1024_256 = torch.nn.Upsample(scale_factor=256/1024, mode="bilinear")
        self.Upsample256_1024 = torch.nn.Upsample(scale_factor=1024/256, mode="bilinear")
        
    def save_log(self, e, loss, syn_img):
        with torch.no_grad():
            print("\riter{}: loss -- {}".format(e, loss.item()))
            save_img(syn_img, "results/" + self.task_name + '/', "blend_{}.png".format(e))
    
    
    def latent_noise(self, latent, strength):
        noise = torch.randn_like(latent) * strength
        return latent + noise
    
    def __call__(self, latent_init):
        
        latent_in, noises, latent_std = G.initcode()
        latent_in.requires_grad = False
        latent_in = torch.cat([latent_init[:, :cfg.blend.mid, :].detach(), latent_in[:, cfg.blend.mid:, :]], dim=1)
        latent_in.requires_grad = True
        
        syn_img, _ = G([latent_in], input_is_latent=True, noise=noises)
        save_img(syn_img, './results/' + self.task_name, 'init_blend.png')
        
        syn_img_d = self.Upsample1024_256(syn_img)
        predictions, HM, FM, BM = get_mask(SegNet, syn_img_d)
        HM, FM = HM.unsqueeze(0).float(), FM.unsqueeze(0).float()
        
        HM_1024 = self.Upsample256_1024(HM)
        FM_1024 = self.Upsample256_1024(FM)
        syn_img_init = syn_img.detach()
        
        opt = optim.Adam([latent_in], lr=cfg.blend.lr, betas=(0.9, 0.999), eps=1e-8)
        
        print(f'Start blend -- {self.task_name}')
        for e in range(cfg.blend.epoch1):
    
            print(f"\rEpoch {e}/{cfg.blend.epoch1}", end="")
            opt.zero_grad()
            noise_strength = latent_std * cfg.styleGAN.noise * max(0, 1 - e / cfg.styleGAN.step / cfg.styleGAN.noise_ramp) ** 2
            latent_n = self.latent_noise(latent_in, noise_strength.item())
            MASK_H = HM_1024 * self.mask_hair_H1024 
            MASK_F = FM_1024 * self.mask_face_F1024 + self.mask_BODY_1024 * (1 - MASK_H)
            syn_img, _ = G([latent_n], input_is_latent=True, noise=noises)
            syn_img = (syn_img + 1) / 2
            
            hair_lpipsloss = self.lpipsloss(syn_img, change_img_TO01(self.image_hair_1024), MASK_H) + self.lpipsloss(syn_img, change_img_TO01(syn_img_init), MASK_H)
            face_lpipsloss = self.lpipsloss(syn_img, change_img_TO01(self.image_face_1024), MASK_F)
            loss = cfg.blend.lamb1_lpipsloss * (hair_lpipsloss + face_lpipsloss) +\
                cfg.blend.lamb1_mseloss_1024_hair * mseloss(syn_img * MASK_H, change_img_TO01(self.image_hair_1024) * MASK_H) + cfg.blend.lamb1_mseloss_1024_hair * mseloss(syn_img * MASK_H, change_img_TO01(syn_img_init) * MASK_H) +\
                cfg.blend.lamb1_mseloss_1024_face * mseloss(syn_img * MASK_F, change_img_TO01(self.image_face_1024) * MASK_F)

            loss.backward()
            opt.step()
            
            if (e + 1) % cfg.blend.print_epoch == 0: 
                self.save_log(e+1, loss, syn_img)
        save_img(syn_img, "./results/" + self.task_name, "final.png")