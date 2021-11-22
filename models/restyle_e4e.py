import torch
import os
import sys

from models.e4e import e4e
from configs.global_config import cfg
from utils.c_utils import *
import torch.nn.functional as F
from utils.common import tensor2im

class Encoder_RestyleE4E():
    def __init__(self):
        net = e4e()
        net.eval()
        net.cuda()
        
        self.avg_image = net(net.latent_avg.unsqueeze(0),
                    input_code=True,
                    randomize_noise=False,
                    return_latents=False,
                    average_code=True)[0].cuda().float().detach()
        self.net = net

    def run_on_batch(self, inputs, avg_image):
        y_hat, latent = None, None
        results_batch = {idx: [] for idx in range(inputs.shape[0])}
        results_latent = {idx: [] for idx in range(inputs.shape[0])}
        for iter in range(cfg.restyle_e4e.n_iters_per_batch):
            if iter == 0:
                avg_image_for_batch = avg_image.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
                x_input = torch.cat([inputs, avg_image_for_batch], dim=1)
            else:
                x_input = torch.cat([inputs, y_hat], dim=1)
    
            y_hat, latent = self.net.forward(x_input,
                                        latent=latent,
                                        randomize_noise=False,
                                        return_latents=True,
                                        resize=False)

            # store intermediate outputs
            for idx in range(inputs.shape[0]):
                results_batch[idx].append(y_hat[idx])
                results_latent[idx].append(latent[idx].cpu().detach().numpy())
            y_hat = self.net.face_pool(y_hat)

        return results_batch, results_latent


    def get_latent_code(self, image: torch.Tensor, image_name: str):
        image = image.cuda().float()
        result_batch, result_latents = self.run_on_batch(image, self.avg_image)
        for i in range(image.shape[0]):
            results = [tensor2im(result_batch[i][iter_idx]) for iter_idx in range(cfg.restyle_e4e.n_iters_per_batch)]

            # for idx, result in enumerate(results):
            #     save_dir = os.path.join(cfg.exp_dir, image_name)
                # os.makedirs(save_dir, exist_ok=True)
                # result.resize((cfg.size, cfg.size)).save(os.path.join(save_dir, "reconstruction_" + str(idx) +'.png'))
        return result_latents