import torch
from image2GAN import ImageReconstruction
from utils.untils import image_reader
import torch.optim as optim
from utils.config import cfg
from loss.loss import LPIPSLoss
from torchvision.utils import save_image
import torch.nn as nn
import os
import warnings
warnings.filterwarnings("ignore")

class Reconstruction:
    def __init__(self):
        self.REC_S1 = ImageReconstruction()
        self.lpipsloss = LPIPSLoss(in_size=1024, out_size=256)
        self.mesloss = nn.MSELoss(reduction="mean")
    def rec(self, image, image_name):

        latent_in, noises = self.REC_S1.reconstruction(image, image_name)
        print('\rstart reconstruction stage2: ', image_name)
        F_init, skip = self.REC_S1.get_layerout(size=32)
        for i in range(len(noises)):
            noises[i] = noises[i].detach()
        if not os.path.exists('./save_image/' + image_name + '/' + "rec_stage1"):
            os.mkdir('./save_image/' + image_name + '/' + "rec_stage1")

        if not os.path.exists('./save_image/' + image_name + '/' + "rec_stage1/param"):
            os.mkdir('./save_image/' + image_name + '/' + "rec_stage1/param")
        path = './save_image/' + image_name + '/' + "rec_stage1/param/"
        torch.save(F_init, path+'F_init.pth')
        torch.save(latent_in, path+'latent_in.pth')
        torch.save(skip, path+'skip.pth')
        if not os.path.exists(path + 'noise'):
            os.mkdir(path + 'noise')
        for j, noise in enumerate(noises[i]):
            torch.save(noises[j], path + 'noise/' + str(j) + '.pth')

        F = F_init.data
        F.requires_grad = True
        S = latent_in.data
        S.requires_grad = True
        
        optim_FS = optim.Adam([F, S], lr=cfg.rec.lr, betas=(0.9, 0.999), eps=1e-8)
        for epoch in range(cfg.rec.fs_epochs):
            print(f'\rEpoch: {epoch}', end='')
            optim_FS.zero_grad()
            loss = 0

            syn_img = self.REC_S1.G.module.mid_start(F, S, noises, skip.detach())
            syn_img = (syn_img + 1.0) / 2.0
            loss += self.lpipsloss(syn_img, image)
            loss += self.mesloss(F, F_init.detach())
            if (epoch+1) % 500 == 0:
                print("\riter{}: loss -- {}".format(epoch + 1, loss.item()))
                if not os.path.exists("save_image/" + image_name + "/rec_stage2"):
                    os.mkdir("save_image/" + image_name + "/rec_stage2")
                save_image(syn_img.clamp(0, 1), "save_image/" + image_name + "/rec_stage2/" + "reconstruct_{}.png".format(epoch + 1))

            loss.backward()
            optim_FS.step()
            optim_FS.zero_grad()
        return F, S


        

if __name__ == '__main__':
    image_path = './images/data/images/67172.jpg'
    img = image_reader(image_path).cuda()
    print(img.shape)
    # reconstruction = Reconstruction()
    # reconstruction.rec(img, 'temp')