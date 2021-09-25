import torch

import sys

from torchvision import transforms
sys.path.append('./')
from networks.image2GAN import ImageReconstruction
from utils.untils import image_reader
import torch.optim as optim
from utils.config import cfg
from loss.loss import LPIPSLoss
from torchvision.utils import save_image
import torch.nn as nn
import os
import warnings
from src.Base import *
warnings.filterwarnings("ignore")

class Reconstruction:
    def __init__(self):
        self.REC_S1 = ImageReconstruction()
        self.lpipsloss = LPIPSLoss(in_size=1024, out_size=256)
    def rec(self, image, image_name):
        print(f"\r\nstart reconstruction -- {image_name} ")
        latent_in, noises = self.REC_S1.reconstruction(image, image_name)
        print('\rstart reconstruction stage2: ', image_name)
        F_init, skip = self.REC_S1.get_layerout(size=32)
        for i in range(len(noises)):
            noises[i] = noises[i].detach()
        if not os.path.exists('./results/' + image_name + "/param"):
            os.mkdir('./results/' + image_name + "/param")

        path = './results/' + image_name + "/param/"

        F = F_init.data
        F.requires_grad = True
        S = latent_in.data
        S.requires_grad = True
        
        optim_FS = optim.Adam([F, S], lr=cfg.rec.lr, betas=(0.9, 0.999), eps=1e-8)
        for epoch in range(cfg.rec.fs_epochs):
            print(f'\rEpoch: {epoch}', end='')
            optim_FS.zero_grad()
            loss = 0

            syn_img = G.mid_start(F, S, noises, skip.detach())
            syn_img = (syn_img + 1.0) / 2.0
            loss += self.lpipsloss(syn_img, image) + cfg.I2SLoss.lamb_mse * mseloss(syn_img, image)
            loss += mseloss(F, F_init.detach())
            if (epoch+1) % 500 == 0:
                print("\riter{}: loss -- {}".format(epoch + 1, loss.item()))
                if not os.path.exists("results/" + image_name + "/rec_stage2"):
                    os.mkdir("results/" + image_name + "/rec_stage2")
                save_image(syn_img.clamp(0, 1), "results/" + image_name + "/rec_stage2/" + "rec_{}.png".format(epoch + 1))

            loss.backward()
            optim_FS.step()
            optim_FS.zero_grad()
        F_all = {}
        for size in [32, 64, 128, 256, 512,1024]:            
            F_all[size] = self.REC_S1.get_layerout(size)
        torch.save(F_all, path+'F_all.pth')
        return F_all, S


        

if __name__ == '__main__':
    image_path = './data/images/0.jpg'

    print(img.shape)
