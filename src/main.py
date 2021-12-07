import sys
import os
import torch

sys.path.append("./")

from utils.c_utils import *
from Base import *
from src.reconstruction import Rec
from src.blend import Blend
from src import genmask
import copy

raw_face = "data/face"
raw_hair = "data/hair"
mask = "data/masks"

# face_image_name = ["0.jpg"]
# hair_image_name = ["00005.png"]
# face_image_name = ["0.jpg", "00013.png", "00020.png", "00066.png", "00103.png"]
# hair_image_name = ["00005.png", "00015.png", "00020.png", "00066.png", "00072.png", "00076.png", "00078.png", "00091.png"]

face_image = sys.argv[1]
hair_image = sys.argv[2]


path = raw_face + '/' + face_image
assert(os.path.exists(path))

path = raw_hair + '/' + hair_image
assert(os.path.exists(path))
    
need_rec = True
need_blend = True
reconstruction = Rec()

task_name = get_image_basename(face_image) + '_' + get_image_basename(hair_image)

os.makedirs(cfg.exp_dir + '/' + task_name, exist_ok=True)
face_image_1024 = load_image(os.path.join(raw_face, face_image), size=1024, normalize=True).cuda()
hair_image_1024 = load_image(os.path.join(raw_hair, hair_image), size=1024, normalize=True).cuda()

save_img(face_image_1024, os.path.join(cfg.exp_dir, task_name), "FaceImage.png")
save_img(hair_image_1024, os.path.join(cfg.exp_dir, task_name), "HairImage.png")

genmask.gen(raw_face + '/' + face_image, mask)
genmask.gen(raw_hair + '/' + hair_image, mask)

face_mask = load_image(os.path.join(mask, get_image_basename(face_image)+'.png'), size=1024, normalize=False).cuda()
hair_mask = load_image(os.path.join(mask, get_image_basename(hair_image)+'.png'), size=1024, normalize=False).cuda()
with torch.no_grad():
    for i in range(3):
        face_mask[:, i, :, :] = (face_mask[:, i, :, :] > 0.5).float()
        hair_mask[:, i, :, :] = (hair_mask[:, i, :, :] > 0.5).float()

FM_1, HM_1 = face_mask[:,1,:,:].unsqueeze(1), face_mask[:, 2, :, :].unsqueeze(1)
FM_2, HM_2 = hair_mask[:,1,:,:].unsqueeze(1), hair_mask[:, 2, :, :].unsqueeze(1)

image_hair, mask_hair = genmask.gen_fullhair(copy.deepcopy(hair_image_1024), copy.deepcopy(HM_2))

M_1, M_2 = face_mask, hair_mask
MASK_H = HM_2
MASK_F = FM_1 * (1 - HM_2)

MASK_BODY = ((1 - M_1.sum(dim=1)) > 0).float()
MASK_H = ((HM_2 + mask_hair - MASK_F - MASK_BODY) > 0.5).float().cuda()
MASK_BG = ((1 - MASK_BODY - MASK_F - MASK_H) > 0.5).float().cuda()

target_img = face_image_1024 * (MASK_F + MASK_BODY) * (1 - MASK_H) + torch.zeros_like(face_image_1024).float().cuda() * MASK_BG + image_hair * MASK_H
target_img_256 = torch.nn.Upsample(scale_factor=256/1024, mode="bilinear")(target_img)

save_img(target_img, os.path.join(cfg.exp_dir, task_name), "target.png")

latent, syn_img = reconstruction.rec(target_img, target_img_256, MASK_F, HM_2, task_name)
save_img(syn_img, os.path.join(cfg.exp_dir, task_name), "rec_final.png", keep=True)
print('reconstruction over, save image to rec_final.png')

blend = Blend(face_image_1024, hair_image_1024, face_mask, hair_mask, task_name)
latent = latent.detach()
F_blend, S_blend = blend(latent)
print('blend over, save image to final.png')

        
        

