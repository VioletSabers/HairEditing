import sys
import os
import torch

sys.path.append("./")

from utils.c_utils import *
from Base import *
from src.reconstruction import Rec
from src import genmask
import copy

raw_face = "data/face"
raw_hair = "data/hair"
mask = "data/masks"

face_image_name = ['0.jpg']
hair_image_name = os.listdir(raw_hair)
hair_image_name.sort()
# hair_image_name = ['00055.png']

for image in face_image_name:
    path = raw_face + '/' + image
    assert(os.path.exists(path))
for image in hair_image_name:
    path = raw_hair + '/' + image
    assert(os.path.exists(path))

reconstruction = Rec()

latent_all = []
for face_image in face_image_name:
    for hair_image in hair_image_name:
        task_name = get_image_basename(face_image) + '_' + get_image_basename(hair_image)
        os.makedirs(cfg.exp_dir + '/' + task_name, exist_ok=True)
        face_image_1024 = load_image(os.path.join(raw_face, face_image), size=1024, normalize=True).cuda()
        face_image_256 = load_image(os.path.join(raw_face, face_image), size=256, normalize=True).cuda()

        hair_image_1024 = load_image(os.path.join(raw_hair, hair_image), size=1024, normalize=True).cuda()
        hair_image_256 = load_image(os.path.join(raw_hair, hair_image), size=256, normalize=True).cuda()

        genmask.gen(raw_face + '/' + face_image, mask)
        genmask.gen(raw_hair + '/' + hair_image, mask)

        face_mask = load_image(os.path.join(mask, get_image_basename(face_image)+'.png'), size=1024, normalize=False).cuda()
        hair_mask = load_image(os.path.join(mask, get_image_basename(hair_image)+'.png'), size=1024, normalize=False).cuda()

        FM_1, HM_1 = face_mask[:,1,:,:].unsqueeze(1), face_mask[:, 2, :, :].unsqueeze(1)
        FM_2, HM_2 = hair_mask[:,1,:,:].unsqueeze(1), hair_mask[:, 2, :, :].unsqueeze(1)

        image_hair, mask_hair = genmask.gen_fullhair(copy.deepcopy(hair_image_1024), copy.deepcopy(HM_2))

        M_1, M_2 = face_mask, hair_mask
        MASK_H = HM_2
        MASK_F = FM_1 * (1 - HM_2)

        MASK_BODY = 1 - M_1.sum(dim=1)
        MASK_H = ((HM_2 + mask_hair - MASK_F - MASK_BODY) > 0.5).float().cuda()
        MASK_BG = ((1 - MASK_BODY - MASK_F - MASK_H) > 0.5).float().cuda()
        target_img = face_image_1024 * (MASK_F + MASK_BODY) * (1 - MASK_H) + torch.ones_like(face_image_1024).float().cuda() * MASK_BG + image_hair * MASK_H
        target_img_256 = torch.nn.Upsample(scale_factor=256/1024, mode="bilinear")(target_img)
        latent, syn_img = reconstruction.rec(target_img, target_img_256, task_name, [face_image_1024, hair_image_1024], [MASK_F, HM_2])

        torch.save(latent, os.path.join(cfg.exp_dir, task_name, "target_latent.pth"))
        save_img(syn_img, os.path.join(cfg.exp_dir, task_name), "rec_result.png")

        save_img(target_img, os.path.join(cfg.exp_dir, task_name), "target.png")
        if len(latent.shape) == 2:
            latent_all.append(latent.unsqueeze(0))
        else:
            latent_all.append(latent)
        torch.save(torch.cat(latent_all, dim=0), "./results/0/latent_all.pth")