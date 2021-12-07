import sys
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from models.segment.graphonomy_inference import get_mask, get_3class
import copy

sys.path.append('./')

from src.Base import *
from utils.c_utils import *
from models.restyle_e4e import Encoder_RestyleE4E
from configs.global_config import cfg
from loss.loss import LPIPSLoss, OrientLoss
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


class Rec():
    def __init__(self):
        if cfg.rec.restyle4e:
            self.encoder = Encoder_RestyleE4E()
        else:
            self.encoder = None
        self.lpipsloss_hair = LPIPSLoss(in_size=1024, out_size=256)
        self.lpipsloss_face = LPIPSLoss(in_size=1024, out_size=256)
        self.Upsample1024_256 = torch.nn.Upsample(scale_factor=256/1024, mode="bilinear")
        self.Upsample256_1024 = torch.nn.Upsample(scale_factor=1024/256, mode="bilinear")
        self.OrientLoss = OrientLoss()
        
        
    def rec(self, image_1024: torch.Tensor, image_256: torch.Tensor, mask_face, mask_hair, image_name: str):
        
        print(f"\rStart Reconstruction -- {image_name} ")

        if cfg.rec.restyle4e:
            latent_in = self.encoder.get_latent_code(image_256, get_image_basename(image_name))
            latent_in = torch.tensor(latent_in[0][-1]).unsqueeze(0).cuda()
            latent_in.requires_grad = True
            latent_in, noises, latent_std = G.initcode(latent_in)
        else:
            latent_in, noises, latent_std = G.initcode()

        syn_img, _ = G([latent_in], input_is_latent=True, noise=noises)
        print('get restyle e4e result, save to init_rec.png')
        save_img(syn_img, "./results/" + get_image_basename(image_name), "init_rec.png")
        
        # w = latent_in[:, 0, :].detach()
        # w.requires_grad = True
        
        w_opt = optim.Adam([latent_in], lr=cfg.rec.lr, betas=(0.9, 0.999), eps=1e-8)
        
        # 优化w
        mask_hair = (mask_hair > 0).float()
        mask_face = (mask_face > 0).float()
        # hair_select = torch.zeros(1, 1, 1024, 1024).cuda()
       
        for e in range(cfg.rec.w_epochs):

            print(f"\rEpoch {e}/{cfg.rec.w_epochs}", end="")
            
            w_opt.zero_grad()
                
            noise_strength = latent_std * cfg.rec.noise * max(0, 1 - e / cfg.rec.step / cfg.rec.noise_ramp) ** 2
            latent_n = latent_noise(latent_in, noise_strength.item())
            syn_img, _ = G([latent_n], input_is_latent=True, noise=noises)

            hair_lpipsloss = self.lpipsloss_hair(syn_img, image_1024, mask_hair)
            face_lpipsloss = self.lpipsloss_face(syn_img, image_1024, mask_face)
            loss = cfg.rec.lamb_lpipsloss_hair * hair_lpipsloss +\
                cfg.rec.lamb_lpipsloss_face * face_lpipsloss +\
                cfg.rec.lamb_mseloss_1024_hair * mseloss(syn_img * mask_hair, image_1024 * mask_hair) +\
                cfg.rec.lamb_mseloss_1024_face * mseloss(syn_img * mask_face, image_1024 * mask_face)
            loss.backward()
            w_opt.step()
            if (e + 1) % cfg.rec.print_epoch == 0:
                print("\riter{}: loss -- {}".format(e + 1, loss.item()))
                save_img(syn_img, "results/" + image_name.split('.')[0] + '/rec', "rec_{}.png".format(e + 1))

        # noises_stage1 = []
        # for noise in noises:
        #     noises_stage1.append(noise.detach())
        # latent_in_stage1 = latent_in.detach()
        
        # latent_in, noises, _ = G.initcode()
        # latent_in.requires_grad = False
        # latent_in = torch.cat([latent_in_stage1[:, :cfg.rec.mid, :], latent_in[:, cfg.rec.mid:, :]], dim=1)
        # latent_in.requires_grad = True
        
        # w_opt = optim.Adam([latent_in], lr=cfg.rec.lr2, betas=(0.9, 0.999), eps=1e-8)
        # syn_img, _ = G([latent_in], input_is_latent=True, noise=noises)
        # save_img(syn_img, "results/" + image_name.split('.')[0], "rec_stage1_final_link.png")
        # image_init = syn_img.detach()
        # syn_img_d = self.Upsample1024_256(syn_img)
        # predictions, HM, FM, BM = get_mask(SegNet, syn_img_d)
        # HM, FM = HM.unsqueeze(0).float(), FM.unsqueeze(0).float()

        # HM_1024 = (self.Upsample256_1024(HM) > 0).float()
        # FM_1024 = (self.Upsample256_1024(FM) > 0).float()
        
        # # 优化style
        # for e in range(total_epoch, total_epoch + cfg.rec.style_epochs):
        #     print(f"\rEpoch {e}", end="")
        #     mask = (torch.rand(1, 1, 1024, 1024) >= 0.95).float().cuda()
        #     w_opt.zero_grad()
                
        #     noise_strength = latent_std * cfg.rec.noise * max(0, 1 - e / cfg.rec.step / cfg.rec.noise_ramp) ** 2
        #     latent_n = latent_noise(latent_in, noise_strength.item())
        #     syn_img, _ = G([latent_n], input_is_latent=True, noise=noises)
            
        #     Other_mask = 1 - dilation(HM_1024, iteration=10)
            
        #     # Orient_synimg = self.OrientLoss.calc(syn_img) * HM_1024 * mask_hair
        #     # Orient_hair = self.OrientLoss.calc(change_img_TO01(image_1024)) * HM_1024 * mask_hair
            
        #     # Orient_synimg = Orient_synimg.view(32, 1024 * 1024)
        #     # Orient_synimg = torch.mm(Orient_synimg, Orient_synimg.t())
            
        #     # Orient_hair = Orient_hair.view(32, 1024 * 1024)
        #     # Orient_hair = torch.mm(Orient_hair, Orient_hair.t())
        #     mask_hair_e = erosion(mask_hair, iteration=20)
        #     hair_styleloss = styleloss(image_1024, syn_img, mask_hair_e, HM_1024)
        #     loss = cfg.rec.lamb_styleloss * hair_styleloss + cfg.rec.lamb_mse * mseloss(image_init * Other_mask, syn_img * Other_mask)
        #     loss = loss + cfg.rec.lamb_mse_hair * mseloss(image_1024 * mask, syn_img * mask)
        #     loss.backward()
        #     w_opt.step()
        #     if (e + 1) % cfg.rec.print_epoch == 0:
        #         print("\riter{}: loss -- {}".format(e + 1, loss.item()))
        #         save_img(syn_img, "results/" + image_name.split('.')[0] + '/rec_stage2', "rec_{}.png".format(e + 1))
        
        return latent_in, syn_img
    
    
if __name__ == "__main__":
    REC = Rec()

    root = "./data/hair"
    image_name = "00005.png"
    out_path = "./results/"+ get_image_basename(image_name)
    
    print('\rfile = ', image_name)
    path = os.path.join(root, image_name)
    image_1024 = load_image(path, size=1024, normalize=True).cuda()
    image_256 = load_image(path, size=256, normalize=True).cuda()

    latent = REC.rec(image_1024, image_256, image_name)
    torch.save(latent, out_path + '/' + get_image_basename(image_name) + '.pth')