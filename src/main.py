import torch
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sys.path.append("./")
from src import genmask
from utils_c import optimizer_utils, image_utils
from datasets.ffhq import process_image
from utils_c import image_utils as ImgU
from src.reconstruction import Reconstruction
from src.blend import Blend
from utils_c.config import cfg
import copy
from torchvision.utils import save_image

print('import ok')

# 正脸 脸全部露出 无头发遮挡 耳朵未露出 
# face_image_name = ["0.jpg", "00020.png", "00013.png", "00066.png", "00103.png"]

# 长直发 无遮挡 正脸
# hair_image_name = ["00001.png", "00013.png", "00048.png", "00055.png", "00059.png", "00065.png", "00095.png", "00105.png", "00116.png", "67172.jpg"]
# 长卷发 无遮挡 正脸
# hair_image_name = ["00005.png", "00015.png", "00020.png", "00066.png", "00072.png", "00076.png", "00078.png", "00091.png"]


image1 = sys.argv[1]
image2 = sys.argv[2]

raw_face = "data/face"
raw_hair = "data/hair"
mask = "data/masks"

# check文件是否存在

path = raw_face + '/' + image1
assert(os.path.exists(path))

path = raw_hair + '/' + image2
assert(os.path.exists(path))

need_rec = True
need_align = True


reconstruction = Reconstruction()


genmask.gen(raw_face + '/' + image1, mask)
genmask.gen(raw_hair + '/' + image2, mask)

if not os.path.exists('./results'):
    os.mkdir('./results')


print('image face = ', image1)
print('image hair = ', image2)

image_files = image_utils.getImagePaths(raw_face, raw_hair, mask, image1, image2)
I_1, M_1, HM_1, H_1, FM_1, F_1 = process_image(
    image_files['I_1_path'], image_files['M_1_path'], size=1024, normalize=1)

I_2, M_2, HM_2, H_2, FM_2, F_2 = process_image(
    image_files['I_2_path'], image_files['M_2_path'], size=1024, normalize=1)

I_1, M_1, HM_1, H_1, FM_1, F_1 = optimizer_utils.make_cuda( 
    [I_1, M_1, HM_1, H_1, FM_1, F_1])
I_2, M_2, HM_2, H_2, FM_2, F_2 = optimizer_utils.make_cuda(
    [I_2, M_2, HM_2, H_2, FM_2, F_2])

I_1, M_1, HM_1, H_1, FM_1, F_1= image_utils.addBatchDim(
    [I_1, M_1, HM_1, H_1, FM_1, F_1])
I_2, M_2, HM_2, H_2, FM_2, F_2 = image_utils.addBatchDim(
    [I_2, M_2, HM_2, H_2, FM_2, F_2])
task_name = image1.split('.')[0] + '_' + image2.split('.')[0]
if not os.path.exists('./results/' + task_name):
    os.mkdir('./results/' + task_name)

I_1, I_2 = ImgU.handle(I_1), ImgU.handle(I_2)

image_hair, mask_hair = genmask.gen_fullhair(copy.deepcopy(I_2), copy.deepcopy(HM_2))


MASK_H = HM_2
MASK_F = FM_1 * (1 - HM_2)
MASK_BODY = 1 - M_1.sum(dim=1)
MASK_H = ((HM_2 + mask_hair - MASK_F - MASK_BODY) > 0.5).float().cuda()
MASK_BG = ((1 - MASK_BODY - MASK_F - MASK_H) > 0.5).float().cuda()
target_img = I_1 * (MASK_F + MASK_BODY) * (1 - MASK_H) + torch.zeros_like(I_1).float().cuda() * MASK_BG + image_hair * MASK_H
save_image(target_img, "results/" + task_name + '/' + "target.png")
# mask_ex = ((mask_target_face + mask_target_hair - HM_2 - FM_1)  > 0.5).float().cuda() * MASK_BG
# hair_meancolor = (MASK_H * I_2).sum(dim=(2, 3)) / MASK_H.sum()
# target_img = target_img * (1 - mask_ex) + hair_meancolor.unsqueeze(-1).unsqueeze(-1) * mask_ex


MASK_t1 = (1 - M_1[:,0,:,:]).unsqueeze(0)
I1_t = I_1 * MASK_t1

MASK_t2 = (1 - M_2[:,0,:,:]).unsqueeze(0)
I2_t = I_2 * MASK_t2


# if need_rec or (not os.path.exists('./results/' + image1.split('.')[0] + '/param/F_all.pth')):
#     reconstruction.rec(I1_t, image1.split('.')[0], is_FS=True)

# if need_rec or (not os.path.exists('./results/' + image2.split('.')[0] + '/param/F_all.pth')):
#     reconstruction.rec(I2_t, image2.split('.')[0], is_FS=True)

if need_rec or (not os.path.exists('./results/' + task_name + '/param/F_all.pth')):
    reconstruction.rec(target_img, task_name, is_FS=False, front=True)

F_F = torch.load('./results/' + image1.split('.')[0] + '/param/F_all.pth')
F_H = torch.load('./results/' + image2.split('.')[0] + '/param/F_all.pth')
F_tar = torch.load('./results/' + task_name + '/param/F_all.pth')


if not os.path.exists('./results/' + task_name):
    os.mkdir('./results/' + task_name)

image_utils.writeImageToDisk(
        [I_1, I_1 * FM_1, I_2, I_2 * HM_2,  I_1 * FM_1 * (1 - HM_2) + I_2 * HM_2], ['Face_image.png', 'Face_select.png', 'Hair_image.png', 'Hair_select.png', 'Target.png'], './results/' + task_name
    )
blend = Blend((I_1, M_1, F_F), (I_2, M_2, F_H), (target_img, torch.cat([MASK_BG, (MASK_F + MASK_BODY) * (1 - MASK_H), MASK_H], dim=1), F_tar), image1, image2)
F_final, S_final, img_final = blend()

torch.save(F_final, './results/' + task_name + '/F_final_' + str(cfg.mid_size) + '.pth')
torch.save(S_final, './results/' + task_name + '/S_final_' + str(cfg.mid_size) + '.pth')
torch.save(img_final, './results/' + task_name + '/syn_final_' + str(cfg.mid_size) + '.pth')

print('ok')
        
