import sys
import os
import torch.nn as nn
sys.path.append("./")
from networks import StyleGAN2
from utils_c.untils import image_reader
import torch
import torch.optim as optim
from torchvision.utils import save_image
import os
from copy import deepcopy
from utils_c.config import cfg
import warnings
from src.Base import *
import copy
warnings.filterwarnings("ignore")

from loss.loss import LPIPSLoss, I2SNoiseLoss

def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise

class ImageReconstruction:
    def __init__(self):
        self.lpipsloss = LPIPSLoss(in_size=1024, out_size=256)
        self.noiseloss = I2SNoiseLoss()
    
    def reconstruction(self, image, image_name, mask_face, mask_hair, front=False):
        print('\rStage1: ', image_name + '.jpg')
        
        if os.path.exists("results/" + image_name.split('.')[0]) == False:
            os.mkdir("results/" + image_name.split('.')[0])
        if os.path.exists("results/" + image_name.split('.')[0] + '/rec_stage1') == False:
            os.mkdir("results/" + image_name.split('.')[0] + '/rec_stage1')
            os.mkdir("results/" + image_name.split('.')[0] + '/rec_stage2')
       
        n_mean_latent = 10000
        
        with torch.no_grad():
            noise_sample = torch.randn(n_mean_latent, 512).cuda()
            latent_out = G.style(noise_sample)

            latent_mean = latent_out.mean(0)
            latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5
        latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(image.shape[0], 1)
        latent_in = latent_in.unsqueeze(1).repeat(1, G.n_latent, 1)
        latent_in.requires_grad = True

        noises_single = G.make_noise()
        noises = []
        for noise in noises_single:
            noises.append(noise.repeat(image.shape[0], 1, 1, 1).normal_())
        for noise in noises:
            noise.requires_grad = True
        
        w_opt = optim.Adam([latent_in], lr=cfg.rec.lr, betas=(0.9, 0.999), eps=1e-8)
        n_opt = optim.Adam(noises, lr=cfg.rec.lr, betas=(0.9, 0.999), eps=1e-8)

        latent_in_init = copy.deepcopy(latent_in.detach().data)
        # 优化w
        for e in range(cfg.rec.w_epochs):
            print(f"\rEpoch {e}", end="")
            w_opt.zero_grad()
            noise_strength = latent_std * cfg.rec.noise * max(0, 1 - e / cfg.rec.step / cfg.rec.noise_ramp) ** 2
            latent_n = latent_noise(latent_in, noise_strength.item())
            syn_img, _ = G([latent_n], input_is_latent=True, noise=noises)
            syn_img = (syn_img + 1.0) / 2.0
            if mask_face is not None:
                loss_face = self.lpipsloss(syn_img, image, mask_face, True) + cfg.I2SLoss.lamb_mse * mseloss(syn_img * mask_face, image * mask_face)
                loss_hair = self.lpipsloss(syn_img, image, mask_hair, True) + cfg.I2SLoss.lamb_mse * mseloss(syn_img * mask_hair, image * mask_hair)
                loss = loss_face + loss_hair
            else:
                loss = self.lpipsloss(syn_img, image) #+ cfg.I2SLoss.lamb_mse * mseloss(syn_img, image)
            loss.backward()
            w_opt.step()
            if (e + 1) % cfg.rec.print_epoch == 0:
                print("\riter{}: loss -- {}".format(e + 1, loss.item()))
                save_image(syn_img.clamp(0, 1), "results/" + image_name.split('.')[0] + '/rec_stage1/' + "rec_{}.png".format(e + 1))
            if front:
                latent_in[9:].data = latent_in_init[9:].data

        # 优化noise
        for e in range(cfg.rec.w_epochs, cfg.rec.w_epochs + cfg.rec.n_epochs):
            print(f"\rEpoch {e}", end="")
            n_opt.zero_grad()
            noise_strength = latent_std * cfg.rec.noise * max(0, 1 - e / cfg.rec.step / cfg.rec.noise_ramp) ** 2
            latent_n = latent_noise(latent_in, noise_strength.item())
            syn_img, _ = G([latent_n], input_is_latent=True, noise=noises)
            syn_img = (syn_img + 1.0) / 2.0
            loss = self.noiseloss(syn_img, image) + cfg.I2SLoss.lamb_mse * mseloss(syn_img, image)
            loss.backward()
            n_opt.step()
            if (e + 1) % cfg.rec.print_epoch == 0:
                print("\riter{}: loss -- {}".format(e + 1, loss.item()))
                save_image(syn_img.clamp(0, 1), "results/" + image_name.split('.')[0] + '/rec_stage1/' + "rec_{}.png".format(e + 1))


        return latent_in, noises
    
    def get_layerout(self, size=32):
        return G.get_layerout(size)

if __name__ == '__main__':
    images_dir = "/data1/gxy_ctl/ffhq/images"

    images_list = os.listdir(images_dir)
    images_list.sort()

    REC = ImageReconstruction()
    cnt = 0
    for i, name in enumerate(images_list):
        if i < 326:
            continue
        print(f'\n-----------------------\n{name}:')
        image = image_reader(images_dir + '/' + name)
        image = image.cuda()

        
        reconstruct_w, reconstruct_noise = REC.reconstruction(image)
        laterout32 = REC.get_layerout(32)

        torch.save([reconstruct_w, laterout32], "/data1/gxy_ctl/ffhq/param/" + name.split('.')[0] + '.pth')


