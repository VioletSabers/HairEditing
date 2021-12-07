from shutil import SameFileError
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
        
        self.lpipsloss_face = LPIPSLoss(in_size=1024, out_size=256)
        self.lpipsloss_hair = LPIPSLoss(in_size=1024, out_size=256)
        self.task_name = task_name
        self.Upsample1024_256 = torch.nn.Upsample(scale_factor=256/1024, mode="bilinear")
        self.Upsample256_1024 = torch.nn.Upsample(scale_factor=1024/256, mode="bilinear")
        
        self.image_hair_256 = self.Upsample1024_256(self.image_hair_1024)
        self.mask_hair_H256 = self.Upsample1024_256(self.mask_hair_H1024)
        
        self.Upsample1024_reshapemid = torch.nn.Upsample(scale_factor=cfg.reshape.mid_size/1024, mode="bilinear")
        self.Upsample256_reshapemid = torch.nn.Upsample(scale_factor=cfg.reshape.mid_size/256, mode="bilinear")
        
        
        
    def save_log(self, e, loss, syn_img):
        with torch.no_grad():
            print("\riter{}: loss -- {}".format(e, loss.item()))
            save_img(syn_img, "results/" + self.task_name + '/', "blend_{}.png".format(e))
    
    def re_save_log(self, e, loss, syn_img):
        with torch.no_grad():
            print("\riter{}: loss -- {}".format(e, loss.item()))
            save_img(syn_img, "results/" + self.task_name + '/', "reshape_{}.png".format(e))
    
    
    def latent_noise(self, latent, strength):
        noise = torch.randn_like(latent) * strength
        return latent + noise
        
    def __call__(self, latent_init: torch.Tensor):
        
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
        
        F_init = G.get_layerout_Latent(cfg.blend.mid_size).detach()
        S = latent_in.detach()
        S.requires_grad = True
        
        F = F_init.detach()
        F.requires_grad = True
        
        opt_FS = optim.Adam([F, S] + noises, lr=cfg.blend.lr, betas=(0.9, 0.999), eps=1e-8)
        opt_W = optim.Adam([S] + noises, lr=cfg.blend.lr * 10, betas=(0.9, 0.999), eps=1e-8)
        
        print('start blend ')
        MASK_H = HM_1024 * self.mask_hair_H1024
        MASK_H = (MASK_H > 0).float()
        MASK_F = (FM_1024 * self.mask_face_F1024 + self.mask_BODY_1024) * (1 - MASK_H)
        MASK_F = (MASK_F > 0).float()
        
        MASK_F_e = erosion(MASK_F, iteration=20)
        mask_hair_e = erosion(self.mask_hair_H1024, iteration=50)
        for e in range(cfg.blend.epoch1):
    
            print(f"\rEpoch {e}/{cfg.blend.epoch1}", end="")
            opt_FS.zero_grad()

            skip = G.get_layerout_Image(cfg.blend.mid_size)
            F_init = G.get_layerout_Latent(cfg.blend.mid_size)
            # syn_img, _ = G([latent_n], input_is_latent=True, noise=noises)
            
            syn_img = G.mid_start(F, S, noise=noises, skip=None, size=cfg.blend.mid_size)
            syn_img_256 = self.Upsample1024_256(syn_img)
            
            
            hair_styleloss = styleloss(self.image_hair_256, syn_img_256, self.mask_hair_H256, HM)
            hair_lpipsloss = self.lpipsloss_hair(syn_img, self.image_hair_1024, mask_hair_e)
            face_lpipsloss = self.lpipsloss_face(syn_img, self.image_face_1024, MASK_F_e)
            loss = cfg.blend.lamb_lpipsloss_hair * hair_lpipsloss + \
                cfg.blend.lamb_lpipsloss_face * face_lpipsloss + \
                cfg.blend.lamb_mseloss_1024_hair * mseloss(syn_img * self.mask_hair_H1024, self.image_hair_1024 * self.mask_hair_H1024) +\
                cfg.blend.lamb_mseloss_1024_face * mseloss(syn_img * MASK_F_e, self.image_face_1024 * MASK_F_e)

            loss = loss + cfg.blend.lamb_w * mseloss(F_init.detach(), F) + cfg.blend.lamb_styleloss * hair_styleloss
            if (e + 1) % cfg.blend.print_epoch == 0: 
                self.save_log(e+1, loss, syn_img)

            loss.backward()
            opt_FS.step()
            
            opt_W.zero_grad()
            syn_img, _ = G([S], noise=noises, input_is_latent=True)
            
            layer_out = G.get_layerout_Latent(cfg.blend.mid_size)
            loss = cfg.blend.lamb_mseloss_mid * mseloss(layer_out, F.detach())
        
            loss.backward()
            opt_W.step()
        
        
        
        syn_img = G.mid_start(F, S, noise=noises, skip=None, size=cfg.blend.mid_size)
        syn_img_d = self.Upsample1024_256(syn_img)
        predictions, HM, FM, BM = get_mask(SegNet, syn_img_d)
        HM, FM = HM.unsqueeze(0).float(), FM.unsqueeze(0).float()
        
        HM_1024 = self.Upsample256_1024(HM)
        FM_1024 = self.Upsample256_1024(FM)
        save_img(syn_img, "./results/" + self.task_name ,"blend_final.png")
        if not os.path.exists('./data/target_mask/' + self.task_name.split('_')[-1] + '.png'):
            save_img(HM_1024, './data/target_mask', self.task_name.split('_')[-1] + '.png', keep=True)
        return F, S
    
    # def reshape(self, latent_S, target_mask):
        # latent_init, noises, latent_std = G.initcode()
        # syn_img, _ = G([latent_S], input_is_latent=True, noise=noises)
        # syn_img_d = self.Upsample1024_256(syn_img)
        # predictions, syn_HM, syn_FM, syn_BM = get_mask(SegNet, syn_img_d)
        # F_init = G.get_layerout_Latent(cfg.reshape.mid_size).detach()
        # skip_init = G.get_layerout_Image(cfg.reshape.mid_size).detach()
        
        # mask_hair_mid = (self.Upsample1024_reshapemid(self.mask_hair_H1024) > 0).float()
        # mask_hair_synmid = (self.Upsample256_reshapemid(syn_HM.float().unsqueeze(0)) > 0).float()
        # mask_bg_synmid = (self.Upsample256_reshapemid(syn_BM.float().unsqueeze(0)) > 0).float()
        
        # mask_hair_targetmid = (self.Upsample1024_reshapemid(target_mask) > 0).float()
        
        # latent_in = latent_S.detach()
        # opt = optim.Adam([latent_in] + noises, lr=cfg.reshape.lr, betas=(0.9, 0.999), eps=1e-8)
        
        # mean_vector_bg = torch.sum(F_init * mask_bg_synmid, dim=(2, 3)) / mask_bg_synmid.sum()
        # mean_vector_hair = torch.sum(F_init * mask_hair_synmid, dim=(2, 3)) / mask_hair_synmid.sum()
        
        # F_hair_add = ((mask_hair_targetmid - mask_hair_synmid) > 0).float()
        # F_hair_add = erosion(F_hair_add, iteration=5)
        # F_hair_minus = ((mask_hair_synmid - mask_hair_targetmid) > 0).float()
        # F_hair_minus = dilation(erosion(F_hair_minus, iteration=5), iteration=10)
        
        # F = F_init.detach() * (1 - F_hair_minus) + F_hair_minus * mean_vector_bg.unsqueeze(-1).unsqueeze(-1)
        # F = F * (1 - F_hair_add) + F_hair_add * mean_vector_hair.unsqueeze(-1).unsqueeze(-1)
        
        # syn_target = G.mid_start(F, latent_in, noise=noises, skip=None, size=cfg.reshape.mid_size)
        # save_img(syn_target)
        
        # for epoch in range(cfg.reshape.epoch):
    
    def reshape(self, F_blend, S_blend, target_mask):
        latent_init, noises, latent_std = G.initcode() 
    
        with torch.no_grad():
            syn_img_W, _ = G([S_blend], input_is_latent=True, noise=noises)
            
            latent_F = G.get_layerout_Latent(cfg.reshape.mid_size)
            latent_skip = G.get_layerout_Image(cfg.reshape.mid_size)
            
            syn_img_init = G.mid_start(F_blend, S_blend, noise=noises, skip = latent_skip.detach(), size=cfg.reshape.mid_size)
            syn_img_d = self.Upsample1024_256(syn_img_init)
        
            predictions, syn_HM, syn_FM, syn_BM = get_mask(SegNet, syn_img_d)

            
            syn_FM_1024 = self.Upsample256_1024(syn_FM.float().unsqueeze(0))
            syn_HM_1024 = self.Upsample256_1024(syn_HM.float().unsqueeze(0))
            

            Face_mask_1024 = ((self.mask_face_F1024 + self.mask_BODY_1024) * (1 - target_mask) > 0).float()
            Hair_mask_1024 = target_mask
            BG_mask_1024 = ((1 - Face_mask_1024 - Hair_mask_1024) > 0).float()
        
            Hair_mask_mid = (self.Upsample1024_reshapemid(Hair_mask_1024) > 0.5).float()
            Face_mask_mid = ((self.Upsample1024_reshapemid(Face_mask_1024) * (1 - Hair_mask_mid)) > 0.5).float()
            BG_mask_mid = (1 - erosion(dilation(Face_mask_mid + Hair_mask_mid, iteration=50), iteration=50) > 0).float()
        
            synHair_mask_mid = (self.Upsample256_reshapemid(syn_HM.unsqueeze(1).float()) > 0.5).float()

            mean_vector_bg = torch.sum(F_blend * BG_mask_mid, dim=(2, 3)) / BG_mask_mid.sum()
            mean_vector_hair = torch.sum(F_blend * synHair_mask_mid, dim=(2, 3)) / synHair_mask_mid.sum()
        F = F_blend.detach()
        F.requires_grad = True   
        S = S_blend.detach()
        S.requires_grad = True
        
        opt_FS = optim.Adam([F, S] + noises, lr=cfg.blend.lr, betas=(0.9, 0.999), eps=1e-8)
        opt_W = optim.Adam([S] + noises, lr=cfg.blend.lr, betas=(0.9, 0.999), eps=1e-8)
        
        
        F_hair_add = ((Hair_mask_mid - synHair_mask_mid) > 0).float()
        # F_hair_add = erosion(F_hair_add, iteration=5)
        F_hair_minus = ((synHair_mask_mid - Hair_mask_mid) > 0).float()
        # F_hair_minus = dilation(erosion(F_hair_minus, iteration=5), iteration=10)
        
        for e in range(cfg.reshape.epoch):
            print(f"\rEpoch {e}", end="")
            opt_FS.zero_grad()
            syn_img = G.mid_start(F, S, noise=noises, skip=None, size=cfg.blend.mid_size)
            
            MASK_H = ((Hair_mask_1024 * syn_HM_1024) > 0).float()
            loss_mse_face = mseloss(syn_img * Face_mask_1024, self.image_face_1024 * Face_mask_1024)
            loss_mse_hair = mseloss(syn_img * MASK_H, self.image_hair_1024 * MASK_H)
            loss_mse_BG = (syn_img * BG_mask_1024).norm()
            loss_lpips_face = self.lpipsloss_face(syn_img, self.image_face_1024, Face_mask_1024)
            # loss_lpips_hair = self.lpipsloss_hair(syn_img, self.image)
            
            loss_hair_styleloss = styleloss(syn_img, syn_img_init, Hair_mask_1024, syn_HM_1024)
            
            
            loss = cfg.reshape.lamb_mseloss_face * loss_mse_face +\
                    cfg.reshape.lamb_mseloss_hair * loss_mse_hair +\
                    cfg.reshape.lamb_lpipsloss * loss_lpips_face  +\
                    cfg.reshape.lamb_styleloss * loss_hair_styleloss +\
                    cfg.reshape.lamb_mseloss_bg * loss_mse_BG
            
            loss.backward()
            opt_FS.step()
            
            if (e + 1) % cfg.reshape.print_epoch == 0: 
                self.re_save_log(e+1, loss, syn_img)