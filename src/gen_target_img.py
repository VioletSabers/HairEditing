import os
import sys
import copy
sys.path.append('./')

from utils.c_utils import *
from src import genmask



file_list = os.listdir('./data/hair')
file_list.sort()

image_hair_all = []
raw_hair = './data/hair'
mask = './data/masks'
for file in file_list:
    hair_image_1024 = load_image(os.path.join(raw_hair, file), size=1024, normalize=True).cuda()
    if os.path.exists('./data/target_hair/' + file):
        image_hair = load_image('./data/target_hair/' + file)
    else:
        genmask.gen(raw_hair + '/' + file, mask)
        hair_mask = load_image(os.path.join(mask, get_image_basename(file)+'.png'), size=1024, normalize=False).cuda()

        FM_2, HM_2 = hair_mask[:,1,:,:].unsqueeze(1), hair_mask[:, 2, :, :].unsqueeze(1)

        hair_mask = load_image(os.path.join(mask, get_image_basename(file)+'.png'), size=1024, normalize=False).cuda()
        image_hair, mask_hair = genmask.gen_fullhair(copy.deepcopy(hair_image_1024), copy.deepcopy(HM_2))
        save_img(image_hair, "./data/target_hair", file)

    image_hair_all.append(image_hair)
torch.save(torch.cat(image_hair_all, dim=0), "./pretrained_model/target_img.pth")
