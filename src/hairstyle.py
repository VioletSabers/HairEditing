import sys
import os
sys.path.append("./")
from utils_c.config import cfg
import scipy.ndimage
from utils_c import optimizer_utils, image_utils
import torch
from datasets.ffhq import process_image
from networks.orientation import orient
import math
import numpy as np
from PIL import Image

class Style:
    def __init__(self, image, mask, need_orient=True):
        assert(image.shape == (1, 3, cfg.size, cfg.size))
        assert(mask.shape == (1, 3, cfg.size, cfg.size))

        mask_hair = mask[0, 0, :, :]
        mask_face = mask[0, 2, :, :]
        
        mask_hair_change = scipy.ndimage.binary_dilation(
                           scipy.ndimage.binary_erosion(mask_hair.cpu(), iterations=5),
                           iterations=5)
        mask_hair_change_torch = torch.from_numpy(mask_hair_change).float().cuda()
        if need_orient:
            cal_orient = orient()


            if image.min() < -0.01:
                fake_image = (image + 1) / 2.0 * 255
            else:
                fake_image = image

            gray = 0.299 * fake_image[:, 0, :, :] + 0.587 * fake_image[:, 1, :, :] + 0.144 * fake_image[:, 2, :, :]
            gray = torch.unsqueeze(gray, 1)

            
            self.orient_tensor, self.confidence_tensor = cal_orient.calOrientation(gray)
            
            confidence_tensor_reshape = (self.confidence_tensor.squeeze() * mask_hair).reshape(cfg.size * cfg.size)
            confidence_tensor_sort, index = torch.sort(confidence_tensor_reshape, descending=True)

            self.image_select = torch.zeros_like(mask_hair)

            self.select_index = []
            for i in range(torch.sum(confidence_tensor_reshape > 0).item() // 10):
                # print(f'\r{str(i)}/{str(torch.sum(confidence_tensor_reshape > 0).item() // 10)}')
                x, y = self.D1toD2(index[i].item())
                self.image_select[x][y] = 1
                self.select_index.append(torch.tensor([x, y]).float())
        
        self.midx, self.midy = image_utils.getImageMiddle(mask_face)
        temp = torch.zeros_like(mask_face)
        temp[self.midx, 0:self.midy] = 1
        self.faceL = (temp * (mask_face > 0.5)).sum()

        temp = torch.zeros_like(mask_face)
        temp[self.midx, self.midy:] = 1
        self.faceR = (temp * (mask_face > 0.5)).sum()

        temp = torch.zeros_like(mask_face)
        temp[0:self.midx, self.midy] = 1
        self.faceT = (temp * (mask_face > 0.5)).sum()
    
    def get_middle(self):
        return torch.tensor([self.midx, self.midy]).float()
    
    def get_faceshape(self):
        return self.faceT, self.faceL, self.faceR
    
    def get_importantNode(self):
        return self.select_index
    
    def get_importantOrientation(self):
        return self.image_select
    
    def get_orientation(self):
        return self.orient_tensor, self.confidence_tensor


    def D1toD2(self, x):
        return (x // cfg.size, x % cfg.size)

    def save(self, img: torch.Tensor):
        while len(img.shape) < 4:
            img = img.unsqueeze(0)
        if img.shape[1] == 3:
            image_utils.writeImageToDisk([img], ['temp_img.png'], './results')
        elif img.shape[1] == 1:
            image_utils.writeMaskToDisk([img], ['temp_mask.png'], './results')
        else:
            assert(0)


if __name__ == '__main__':
    raw = "data/images"
    mask = "data/masks"
    background = "data/backgrounds"
    softmask = "data/softmasks"
    image1 = "0.jpg"
    image_files = image_utils.getImagePaths(raw, mask, background, image1)
    I_1, M_1, HM_1, H_1, FM_1, F_1 = process_image(
        image_files['I_1_path'], image_files['M_1_path'], size=1024, normalize=1)

    I_1, M_1, HM_1, H_1, FM_1, F_1 = optimizer_utils.make_cuda(
        [I_1, M_1, HM_1, H_1, FM_1, F_1])

    I_1, M_1, HM_1, H_1, FM_1, F_1= image_utils.addBatchDim(
        [I_1, M_1, HM_1, H_1, FM_1, F_1])
    
    HairStyle = Style(I_1, M_1)

    print(HairStyle.get_middle())
    print(HairStyle.get_faceshape())
    
