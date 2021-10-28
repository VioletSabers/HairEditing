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
from src.blend import Test
from utils_c.config import cfg
from utils_c import image_utils as ImgU

print('import ok')

# 正脸 脸全部露出 无头发遮挡 耳朵未露出 
# face_image_name = ["0.jpg", "00020.png", "00013.png", "00066.png", "00103.png"]

# 长直发 无遮挡 正脸
# hair_image_name = ["00001.png", "00013.png", "00048.png", "00055.png", "00059.png", "00065.png", "00095.png", "00105.png", "00116.png", "67172.jpg"]
# 长卷发 无遮挡 正脸
# hair_image_name = ["00005.png", "00015.png", "00020.png", "00066.png", "00072.png", "00076.png", "00078.png", "00091.png", "00105.png"]

raw_face = "data/face"
raw_hair = "data/hair"
mask = "data/masks"

# check文件是否存在

need_rec = False
need_align = False

image1 = '00091.png'

reconstruction = Reconstruction()

genmask.gen(raw_hair + '/' + image1, mask)

if not os.path.exists('./results'):
    os.mkdir('./results')

print('image face = ', image1)

image_files = image_utils.getImagePaths(raw_hair, raw_hair, mask, image1, image1)
I_1, M_1, HM_1, H_1, FM_1, F_1 = process_image(
    image_files['I_1_path'], image_files['M_1_path'], size=1024, normalize=1)

I_1, M_1, HM_1, H_1, FM_1, F_1 = optimizer_utils.make_cuda( 
    [I_1, M_1, HM_1, H_1, FM_1, F_1])

I_1, M_1, HM_1, H_1, FM_1, F_1= image_utils.addBatchDim(
    [I_1, M_1, HM_1, H_1, FM_1, F_1])

I_1 = ImgU.handle(I_1)
MASK_t1 = (1 - M_1[:,0,:,:]).unsqueeze(0)
I1_t = I_1 * MASK_t1

if need_rec or (not os.path.exists('./results/' + image1.split('.')[0] + '/param/F_all.pth')):
    reconstruction.rec(I1_t, image1.split('.')[0])

F_F = torch.load('./results/' + image1.split('.')[0] + '/param/F_all.pth')

test = Test(I_1, M_1, F_F, image1)
test()
print('ok')
    
