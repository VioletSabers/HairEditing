#%%
import sys
sys.path.append('./')
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image
from torchvision import datasets, transforms
import torch.nn as nn
from utils_c import image_utils as ImgU


import sys
sys.path.append('./')
from networks import deeplab_xception_transfer
from networks.graphonomy_inference import get_mask
import torch.nn.functional as F
from utils_c import image_utils
import numpy as np

adaptor_root_dir = '/data1/guoxuyang/myWorkSpace/hair_editing'
sys.path.append(adaptor_root_dir)
sys.path.append(os.path.join(adaptor_root_dir, 'external_code/face_3DDFA'))

from mask_adaptor import wrap_for_FFHQ, write_rgb, get_parsing_show, wrap_by_path
from global_value_utils import PARSING_LABEL_LIST, PARSING_COLOR_LIST

#%%

def gen(root_path, mask_path):
    
    size = 1024
    transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    img = Image.open(root_path).convert("RGB")
    data = transform(img)
    SegNet = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(
            n_classes=20,
            hidden_layers=128,
            source_classes=7,
        )
    state_dict = torch.load('./pretrain_model/inference.pth')
    SegNet.load_source_model(state_dict)
    SegNet = SegNet.cuda()
    SegNet.eval()

    downsample = nn.Upsample(scale_factor=256/1024, mode='bilinear').cuda()
    cnt = 0
    
    data = data.unsqueeze(0).cuda()
    img_d = downsample(data)
    predictions, HM, FM, BG = get_mask(SegNet, img_d)

    HM = HM.unsqueeze(1)
    FM = FM.unsqueeze(1)
    BG = BG.unsqueeze(1)
    mask = torch.cat([BG, FM, HM], dim=1)
    name = root_path.split('/')[-1].split('.')[0]
    image_utils.writeImageToDisk(
            [mask[0].unsqueeze(0)], [name + '.png'], mask_path
        )
    print('genmask ' + root_path.split('/')[-1] + ' ok')

HAIR_IDX = PARSING_LABEL_LIST.index('hair')
FACE_IDX = [
    PARSING_LABEL_LIST.index('nose'),
    PARSING_LABEL_LIST.index('l_eye'),
    PARSING_LABEL_LIST.index('r_eye'),
    PARSING_LABEL_LIST.index('l_brow'),
    PARSING_LABEL_LIST.index('r_brow'),
    PARSING_LABEL_LIST.index('l_ear'),
    PARSING_LABEL_LIST.index('r_ear'),
    PARSING_LABEL_LIST.index('mouth'),
    PARSING_LABEL_LIST.index('u_lip'),
    PARSING_LABEL_LIST.index('l_lip'),
    PARSING_LABEL_LIST.index('skin_other')
]
SKIN_IDX = PARSING_LABEL_LIST.index('skin_other')

def gen_targetmask(output_dir, face_dir, hair_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    result_parsing, other_result = wrap_by_path(hair_dir, face_dir, wrap_temp_folder=output_dir, fully=True)
    result_show = get_parsing_show(result_parsing)
    write_rgb(os.path.join(output_dir, 'result_show.png'), result_show)

    hair_mask = (result_parsing == HAIR_IDX).astype('uint8')
    skin_mask = (result_parsing == SKIN_IDX).astype('uint8')
    face_mask = np.zeros_like(result_parsing).astype('uint8')
    for index in FACE_IDX:
        face_mask = face_mask + (result_parsing == index).astype('uint8')
    

    face_mask, hair_mask, skin_mask =  torch.from_numpy(face_mask).unsqueeze(0).unsqueeze(0), \
                                       torch.from_numpy(hair_mask).unsqueeze(0).unsqueeze(0), \
                                       torch.from_numpy(skin_mask).unsqueeze(0).unsqueeze(0)
    face_mask, hair_mask, skin_mask =   nn.Upsample(scale_factor=2)(face_mask), \
                                        nn.Upsample(scale_factor=2)(hair_mask), \
                                        nn.Upsample(scale_factor=2)(skin_mask),
    return face_mask.cuda(), hair_mask.cuda(), skin_mask.cuda()

def gen_fullhair(image_hair, mask_hair):
    assert(image_hair.shape == (1, 3, 1024, 1024))
    assert(mask_hair.shape == (1, 1, 1024, 1024))

    with torch.no_grad():
        mask_hair = ImgU.erosion(mask_hair, iteration=5)

        label = torch.tensor(range(512)).cuda()

        for i in range(1024):
            if mask_hair[0, 0, i, :512].sum() == 0:
                continue
            L = label[(mask_hair[0, 0, i, :512] * label) != 0].min()
            R = label[(mask_hair[0, 0, i, :512] * label) != 0].max() + 1
            if (R == 512):
                continue
            select = image_hair[0, :, i, L:R]
            select = select.unsqueeze(1).repeat(1, 512, 1).reshape(3, 512*(R-L))
            mask_hair[0, 0, i, L:512] = 1
            image_hair[0, :, i, L:512] = select[:,:512-L]
        
        label = label + 512
        for i in range(1024):
            if mask_hair[0, 0, i, 512:].sum() == 0:
                continue
            L = label[(mask_hair[0, 0, i, 512:] * label) != 0].min()
            R = label[(mask_hair[0, 0, i, 512:] * label) != 0].max() + 1
            if (L == 512):
                continue
            select = image_hair[0, :, i, L:R]
            select = select.unsqueeze(1).repeat(1, 512, 1).reshape(3, 512*(R-L))
            mask_hair[0, 0, i, 512:L+1] = 1
            image_hair[0, :, i, 512:L+1] = select[:,-(L+1-512):]
    return image_hair, mask_hair

def gen_targetmask_hair(image_hair, mask_hair, target_mask):
    assert(image_hair.shape == (1, 3, 1024, 1024))
    assert(mask_hair.shape == (1, 1, 1024, 1024))
    assert(target_mask.shape == (1, 1, 1024, 1024))
    with torch.no_grad():
        mask_hair_E = ImgU.erosion(mask_hair, iteration=10)
        
        label = torch.tensor(range(512)).cuda()

        no_hair = True
        for i in range(1024):
            if target_mask[0, 0, i, :512].sum() == 0 or i < 32 or target_mask[0, 0, i-32, 512:].sum() == 0:
                continue
            if mask_hair_E[0, 0, i, :512].sum() == 0 and no_hair:
                continue
            else:
                if mask_hair_E[0, 0, i, :512].sum() == 0:
                    L = label[(target_mask[0, 0, i-32, :512] * label) != 0].min()
                    R = label[(target_mask[0, 0, i-32, :512] * label) != 0].max() + 1
                    row = i - 32
                else:
                    L = label[(mask_hair_E[0, 0, i, :512] * label) != 0].min()
                    R = label[(mask_hair_E[0, 0, i, :512] * label) != 0].max() + 1
                    if R - L < 10:
                        L = label[(target_mask[0, 0, i-32, :512] * label) != 0].min()
                        R = label[(target_mask[0, 0, i-32, :512] * label) != 0].max() + 1

                        row = i - 32
                    else:
                        row = i
            L, R = int(L.item()), int(R.item())
            select = image_hair[0, :, row, L:R]


            no_hair = False
            L_tar = int(label[(target_mask[0, 0, i, :512] * label) != 0].min().item())
            R_tar = int(label[(target_mask[0, 0, i, :512] * label) != 0].max().item()) + 1



            tar = F.interpolate(select.unsqueeze(0), size=(R_tar - L_tar))
            image_hair[0, :, i, L_tar:R_tar] = tar.squeeze()
        
        no_hair = True
        label = label + 512
        for i in range(1024):
            if target_mask[0, 0, i, 512:].sum() == 0 or i < 32 or target_mask[0, 0, i-32, 512:].sum() == 0:
                continue
            if mask_hair_E[0, 0, i, 512:].sum() == 0 and no_hair:
                continue
            else:
                if mask_hair_E[0, 0, i, 512:].sum() == 0:
                    L = label[(target_mask[0, 0, i-32, 512:] * label) != 0].min()
                    R = label[(target_mask[0, 0, i-32, 512:] * label) != 0].max() + 1
                    row = i - 32
                else:
                    L = label[(mask_hair_E[0, 0, i, 512:] * label) != 0].min()
                    R = label[(mask_hair_E[0, 0, i, 512:] * label) != 0].max() + 1
                    row = i
                    if R - L < 10:
                        L = label[(target_mask[0, 0, i-32, 512:] * label) != 0].min()
                        R = label[(target_mask[0, 0, i-32, 512:] * label) != 0].max() + 1
                        row = i - 32
            L, R = int(L.item()), int(R.item())
            no_hair = False
            L_tar = int(label[(target_mask[0, 0, i, 512:] * label) != 0].min().item())
            R_tar = int(label[(target_mask[0, 0, i, 512:] * label) != 0].max().item()) + 1

            select = image_hair[0, :, row, L:R]

            tar = F.interpolate(select.unsqueeze(0), size=(R_tar - L_tar))
            image_hair[0, :, i, L_tar:R_tar] = tar.squeeze()
    return image_hair

if __name__ == '__main__':
    image = '00091.png'
    mask_path = 'data/masks/'
    image_path = 'data/hair/'
    size = 1024
    transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    img = Image.open(image_path + image).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0)

    mask = Image.open(mask_path + image).convert("RGB")
    mask = transform(mask)
    mask_hair = mask[2, :, :].unsqueeze(0).unsqueeze(0)

    gen_fullhair(img, mask_hair)

