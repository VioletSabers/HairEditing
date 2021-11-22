"""
This file defines the core research contribution
"""
import math
import torch
from torch import nn
import sys
from Base import G
from utils.c_utils import *

sys.path.append("./")

from models.encoders import restyle_e4e_encoders
from configs.global_config import cfg
import torch.nn.functional as F


class e4e(nn.Module):

    def __init__(self):
        super(e4e, self).__init__()
        self.n_styles = int(math.log(cfg.size, 2)) * 2 - 2
        # Define architecture
        self.encoder = restyle_e4e_encoders.ProgressiveBackboneEncoder(50, 'ir_se', self.n_styles)
        self.decoder = G

        # Load weights if needed
        print(f'Loading ReStyle e4e from checkpoint: {cfg.modelpath.restyle_e4e}')
        param = torch.load(cfg.modelpath.restyle_e4e, map_location='cpu')
        self.encoder.load_state_dict(param, strict=False)
        self.latent_avg = torch.load(cfg.modelpath.latent_avg)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((cfg.restyle_e4e.size, cfg.restyle_e4e.size))

        self.encoder = self.encoder.cuda()
        self.latent_avg = self.latent_avg.cuda()

    def forward(self, x, latent=None, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None, average_code=False, input_is_full=False):
        if input_code:
            codes = x
        else:
            codes = self.encoder(x)
            # residual step
            if x.shape[1] == 6 and latent is not None:
                # learn error with respect to previous iteration
                codes = codes + latent
            else:
                # first iteration is with respect to the avg latent code
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        if average_code:
            input_is_latent = True
        else:
            input_is_latent = (not input_code) or (input_is_full)

        images, result_latent = self.decoder([codes],
                                             input_is_latent=input_is_latent,
                                             randomize_noise=randomize_noise,
                                             return_latents=return_latents)

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent
        else:
            return images