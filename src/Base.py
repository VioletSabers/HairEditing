from configs.global_config import cfg
import torch
from models.segment import deeplab_xception_transfer
from models.stylegan2 import model
from loss.loss import LPIPSLoss, StyleLoss, SegLoss
import torch.nn.functional as F

# 构建生成网络
pretrain_path = cfg.modelpath.styleGAN2
G = model.Generator(cfg.styleGAN.size, cfg.styleGAN.dimention)
G.load_state_dict(torch.load(pretrain_path), strict=False)
G = G.cuda()
G.eval()

SegNet = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(
    n_classes=20,
    hidden_layers=128,
    source_classes=7,
)

state_dict = torch.load(cfg.modelpath.segment)
SegNet.load_source_model(state_dict)
SegNet = SegNet.cuda()
SegNet.eval()

styleloss = StyleLoss()
segloss = SegLoss()
mseloss = torch.nn.MSELoss(reduction='mean')

        