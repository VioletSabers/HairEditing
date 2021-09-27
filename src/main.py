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


image_name = ["0.jpg", "00018.jpg", "00761.jpg", "01012.jpg", "02602.jpg", "10446.jpg" , "67172.jpg"]

# image1 = sys.argv[1]
# image2 = sys.argv[2]
reconstruction = Reconstruction()
for image1 in image_name:
    for image2 in image_name:
        if image1 == image2:
            continue
        raw = "data/images"
        mask = "data/masks"

        if not os.path.exists(mask + '/' + image1.split('.')[0] + '.png'):
            genmask.gen(raw + '/' + image1, mask)
        if not os.path.exists(mask + '/' + image2.split('.')[0] + '.png'):
            genmask.gen(raw + '/' + image2, mask)

        if not os.path.exists('./results'):
            os.mkdir('./results')
        file = os.listdir('./results')

        # if ((image1.split('.')[0] + '_' + image2.split('.')[0]) in file):
        #     continue
        image_files = image_utils.getImagePaths(raw, mask, image1, image2)
        print('image face = ', image1)
        print('image hair = ', image2)
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
        

        if not os.path.exists('./results/' + image1.split('.')[0] + '/param/F_all.pth'):
            reconstruction.rec(I_1, image1.split('.')[0])
        F_F = torch.load('./results/' + image1.split('.')[0] + '/param/F_all.pth')

        if not os.path.exists('./results/' + image2.split('.')[0] + '/param/F_all.pth'):
            reconstruction.rec(I_2, image2.split('.')[0])
        F_H = torch.load('./results/' + image2.split('.')[0] + '/param/F_all.pth')

        task_name = image1.split('.')[0] + '_' + image2.split('.')[0]
        if not os.path.exists('./results/' + task_name):
            os.mkdir('./results/' + task_name)

        blend_name = image1.split('.')[0] + '_' + image2.split('.')[0]

        image_utils.writeImageToDisk(
                [I_1, I_1 * FM_1, I_2, I_2 * HM_2,  I_1 * FM_1 * (1 - HM_2) + I_2 * HM_2], ['Face_image.png', 'Face_select.png', 'Hair_image.png', 'Hair_select.png', 'Target.png'], './results/' + blend_name
            )
        rec = Blend((I_1, M_1, F_F), (I_2, M_2, F_H), None, None, image1, image2)
        if not os.path.exists('./results/' + blend_name + '/F_final_' + str(cfg.mid_size) + '.pth'):
            F_final, S_final, syn_final = rec()
            F_final = F_final.data
            S_final = S_final.data
            F_final.requires_grad = False
            S_final.requires_grad = False
            torch.save(F_final, './results/' + blend_name + '/F_final_' + str(cfg.mid_size) + '.pth')
            torch.save(S_final, './results/' + blend_name + '/S_final_' + str(cfg.mid_size) + '.pth')
            torch.save(syn_final, './results/' + blend_name + '/syn_final_' + str(cfg.mid_size) + '.pth')

        F_final = torch.load('./results/' + blend_name + '/F_final_' + str(cfg.mid_size) + '.pth')
        S_final = torch.load('./results/' + blend_name + '/S_final_' + str(cfg.mid_size) + '.pth')
        syn_final = torch.load('./results/' + blend_name + '/syn_final_' + str(cfg.mid_size) + '.pth')
        rec.blend_final(F_final, S_final, syn_final)

        del rec
print('ok')
