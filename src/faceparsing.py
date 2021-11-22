import sys
import os

from numpy import fmax
from  utils import optimizer_utils, image_utils
import torch
from torchvision.transforms import transforms
import scipy.ndimage
from datasets.ffhq import process_image

def add_batch(image: torch.Tensor):
    while len(image.shape) < 4:
        image = image.unsqueeze(0)
    return image

class Parsing:
    def __init__(self, size=1024):
        self.size = size
        self.resize = transforms.Resize((size, size))

    def get_Face_Noface(self, face_mask: torch.Tensor, hair_mask: torch.Tensor):
        if len(face_mask.shape) == 3:
            face_mask.unsqueeze(0)
        if len(hair_mask.shape) == 3:
            hair_mask.unsqueeze(0)
        assert(face_mask.shape == (1, 3, self.size, self.size))
        assert(hair_mask.shape == (1, 3, self.size, self.size))

        FM_face = face_mask[0][2]
        HM_face = face_mask[0][0]

        FM_delate = scipy.ndimage.binary_dilation(
            FM_face.cpu(), iterations=5
        )

        HM_delate = scipy.ndimage.binary_dilation(
            HM_face.cpu(), iterations=5
        )

        FM_delate = torch.from_numpy(FM_delate).float().cuda()
        HM_delate = torch.from_numpy(HM_delate).float().cuda()
        # bg = (FM_delate - FM_face) * (1 - HM_delate)
        bg = torch.ones_like(FM_face) - FM_delate



        return torch.cat([torch.zeros((1, 1, self.size, self.size)).cuda(),
                          torch.zeros((1, 1, self.size, self.size)).cuda(),
                          torch.zeros((1, 1, self.size, self.size)).cuda(),
                          add_batch(bg)], 
                          dim=1
                          )
    def get_NoHair(self, face_mask, hair_mask):
        if len(face_mask.shape) == 3:
            face_mask.unsqueeze(0)
        if len(hair_mask.shape) == 3:
            hair_mask.unsqueeze(0)
        assert(face_mask.shape == (1, 1, self.size, self.size))
        assert(hair_mask.shape == (1, 1, self.size, self.size))
        
        HM_hair = hair_mask[0][0]
        FM_hair = face_mask[0][0]
        HM_delate = scipy.ndimage.binary_dilation(
            HM_hair.cpu(), iterations=5
        )

        FM_delate = scipy.ndimage.binary_dilation(
            FM_hair.cpu(), iterations=5
        )


        HM_delate = torch.from_numpy(HM_delate).float().cuda()
        FM_delate = torch.from_numpy(FM_delate).float().cuda()
        bg = ((torch.ones_like(HM_hair) - HM_delate - FM_delate) > 0.5)

        bg_erode = scipy.ndimage.binary_dilation(
            bg.float().cpu(), iterations=3
        )
        bg_erode = torch.from_numpy(bg_erode).float().cuda().unsqueeze(0).unsqueeze(0)
        return bg_erode

if __name__ == '__main__':

    raw = "data/images"
    mask = "data/masks"
    background = "data/backgrounds"
    softmask = "data/softmasks"
    image1 = "02602.jpg"
    image2 = "08244.jpg"
    image_files = image_utils.getImagePaths(raw, mask, background, image1, image2)

    I_1, M_1, HM_1, H_1, FM_1, F_1 = process_image(
        image_files['I_1_path'], image_files['M_1_path'], size=1024, normalize=1)

    I_2, M_2, HM_2, H_2, FM_2, F_2 = process_image(
        image_files['I_2_path'], image_files['M_2_path'], size=1024, normalize=1)


    I_1, M_1, HM_1, H_1, FM_1, F_1 = optimizer_utils.make_cuda(
        [I_1, M_1, HM_1, H_1, FM_1, F_1])
    I_2, M_2, HM_2, H_2, FM_2, F_2 = optimizer_utils.make_cuda(
        [I_2, M_2, HM_2, H_2, FM_2, F_2])
    
    I_1, M_1, HM_1, H_1, FM_1, F_1 = image_utils.addBatchDim(
        [I_1, M_1, HM_1, H_1, FM_1, F_1])
    I_2, M_2, HM_2, H_2, FM_2, F_2 = image_utils.addBatchDim(
        [I_2, M_2, HM_2, H_2, FM_2, F_2])
    
    parsing = Parsing(1024)
    mask = parsing.get_Face_Noface(M_1, M_2)

    mask[0][0] = mask[0][3]
    print(mask[0])
    image_utils.writeImageToDisk(
            [mask[:,0:3,:,:].clone()], [f'temp.png'], './results'
        )


