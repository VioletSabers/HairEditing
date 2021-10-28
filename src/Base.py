from utils_c.config import cfg
import torch
from networks import StyleGAN2
from networks import deeplab_xception_transfer
from loss.loss import LPIPSLoss, StyleLoss, SegLoss


SegNet = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(
    n_classes=20,
    hidden_layers=128,
    source_classes=7,
)

state_dict = torch.load(cfg.modelpath.segment)
SegNet.load_source_model(state_dict)
SegNet = SegNet.cuda()
SegNet.eval()


# 构建生成网络
pretrain_path = cfg.modelpath.styleGAN2
G = StyleGAN2.Generator(cfg.styleGAN.size, cfg.styleGAN.dimention)
G.load_state_dict(torch.load(pretrain_path)["g_ema"], strict=False)
G = G.cuda()
G.eval()


styleloss = StyleLoss()
segloss = SegLoss()
mseloss = torch.nn.MSELoss(reduction='mean')

class BaseClass:

    def __init__(self):
        self.resample_list = {}
    
    @staticmethod
    def initnoise(latent_in_m: torch.Tensor, noises_m):
        n_mean_latent = 10000
        with torch.no_grad():
            noise_sample = torch.randn(n_mean_latent, 512).cuda()
            latent_out = G.style(noise_sample)
            latent_out: torch.Tensor
            
            latent_mean = latent_out.mean(0)
            latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5
        if latent_in_m is None:
            
            latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(1, 1)
            latent_in = latent_in.unsqueeze(1).repeat(1, G.n_latent, 1)
            latent_in: torch.Tensor

            latent_in.requires_grad = True
        else:
            latent_in = latent_in_m
            latent_in.requires_grad = True
        
        if noises_m is None:

            noises_single = G.make_noise()
            noises = []
            for noise in noises_single:
                noises.append(noise.repeat(1, 1, 1, 1).normal_())
            for noise in noises:
                noise.requires_grad = True
        else:
            noises = noises_m
            for noise in noises:
                noise.requires_grad = True
        return latent_in, noises, latent_std

    def resample(self, data, insize, outsize, mode='bilinear'):
        assert(insize > 0 and outsize > 0)
        if (insize, outsize) in self.resample_list:
            return self.resample_list[(insize, outsize)](data)
        else:
            self.resample_list[(insize, outsize)] = torch.nn.Upsample(scale_factor=outsize / insize, mode=mode)
            return self.resample_list[(insize, outsize)] (data)

        