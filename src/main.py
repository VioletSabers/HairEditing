import torch
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sys.path.append("./")
from src import genmask
from utils import optimizer_utils, image_utils
from datasets.ffhq import process_image
from utils import image_utils as ImgU
from src.reconstruction import Reconstruction
from src.blend import Blend
from utils.config import cfg

print('import ok')

# 正脸 脸全部露出 无头发遮挡 耳朵未露出 
# face_image_name = ["0.jpg", "00020.png", "00013.png", "00066.png", "00103.png"]

# 长直发 无遮挡 正脸
# hair_image_name = ["00001.png", "00013.png", "00048.png", "00055.png", "00059.png", "00065.png", "00095.png", "00105.png", "00116.png", "67172.jpg"]
# 长卷发 无遮挡 正脸
# hair_image_name = ["00005.png", "00015.png", "00020.png", "00066.png", "00072.png", "00076.png", "00078.png", "00091.png", "00105.png"]


face_image_name = ['0.jpg']
hair_image_name = ['00091.png']



raw_face = "data/face"
raw_hair = "data/hair"
mask = "data/masks"

# check文件是否存在
for image in face_image_name:
    path = raw_face + '/' + image
    assert(os.path.exists(path))
for image in hair_image_name:
    path = raw_hair + '/' + image
    assert(os.path.exists(path))

need_rec = False
need_align = True


reconstruction = Reconstruction()
for image1 in face_image_name:
    for image2 in hair_image_name:

        if not os.path.exists(mask + '/' + image1.split('.')[0] + '.png'):
            genmask.gen(raw_face + '/' + image1, mask)
        if not os.path.exists(mask + '/' + image2.split('.')[0] + '.png'):
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

        I_1, I_2 = ImgU.handle(I_1), ImgU.handle(I_2)


        if need_rec or (not os.path.exists('./results/' + image1.split('.')[0] + '/param/F_all.pth')):
            reconstruction.rec(I_1, image1.split('.')[0])
        F_F = torch.load('./results/' + image1.split('.')[0] + '/param/F_all.pth')

        if need_rec or (not os.path.exists('./results/' + image2.split('.')[0] + '/param/F_all.pth')):
            reconstruction.rec(I_2, image2.split('.')[0])
        F_H = torch.load('./results/' + image2.split('.')[0] + '/param/F_all.pth')

        task_name = image1.split('.')[0] + '_' + image2.split('.')[0]
        if not os.path.exists('./results/' + task_name):
            os.mkdir('./results/' + task_name)

        blend_name = image1.split('.')[0] + '_' + image2.split('.')[0]

        image_utils.writeImageToDisk(
                [I_1, I_1 * FM_1, I_2, I_2 * HM_2,  I_1 * FM_1 * (1 - HM_2) + I_2 * HM_2], ['Face_image.png', 'Face_select.png', 'Hair_image.png', 'Hair_select.png', 'Target.png'], './results/' + blend_name
            )
        blend = Blend((I_1, M_1, F_F), (I_2, M_2, F_H), None, image1, image2)
        if need_align or (not os.path.exists('./results/' + blend_name + '/F_final_' + str(cfg.mid_size) + '.pth')):
            F_final, S_final, syn_final = blend()
            torch.save(F_final, './results/' + blend_name + '/F_final_' + str(cfg.mid_size) + '.pth')
            torch.save(S_final, './results/' + blend_name + '/S_final_' + str(cfg.mid_size) + '.pth')
            torch.save(syn_final, './results/' + blend_name + '/syn_final_' + str(cfg.mid_size) + '.pth')

        F_final = torch.load('./results/' + blend_name + '/F_final_' + str(cfg.mid_size) + '.pth')
        S_final = torch.load('./results/' + blend_name + '/S_final_' + str(cfg.mid_size) + '.pth')
        syn_final = torch.load('./results/' + blend_name + '/syn_final_' + str(cfg.mid_size) + '.pth')

        # target_color = torch.tensor([[0.704, 0.187, 0.897]]).float().cuda()

        target_color = (I_2 * HM_2).sum(dim=(2, 3)) / HM_2.sum()
        target_color = target_color.unsqueeze(-1).unsqueeze(-1)
        blend.blend_final(F_final, S_final, syn_final, ImgU.RGBtoHSV(target_color))
        # blend.blend_final(F_final, S_final, syn_final, None)

        print('ok')
    
