import torch.nn as nn
from torch.nn.modules.loss import MSELoss
import sys

from torch.optim.optimizer import Optimizer
sys.path.append('./')
from networks.VGG16 import VGG16_perceptual
from torch.utils.data import Dataset, DataLoader
from utils.untils import image_reader
import torch
import os
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")


class Encoder(nn.Module):
    def __init__(self, in_size=1024, out_size=256):
        super(Encoder, self).__init__() 
        self.perceptual = VGG16_perceptual(requires_grad=True)
        self.MSE_loss = nn.MSELoss(reduction="mean")
        self.upsample = nn.Upsample(scale_factor=out_size / in_size)
        self.insize = in_size
        self.outsize = out_size
    def forward(self, img, param):
        assert(img.shape[1:] == (3, 1024, 1024))
        assert(param.shape[1:] == (512, 32, 32))
        img_d = self.upsample(img)
        syn0, syn1, syn2, syn3 = self.perceptual(img_d) 
        loss = self.MSE_loss(syn3, param)
        return loss

class MyDataset(Dataset):
    def __init__(self, image_path, param_path):
        image_list = os.listdir(image_path)
        param_list = os.listdir(param_path)
        param_list.sort()
        self.image, self.param = [], []
        for name_param in param_list:
            print(f'\rloading {name_param}', end='')
            name = name_param.split('.')[0]
            image = image_reader(image_path + '/' + name + '.png').squeeze(0)
            w, (p, skip) = torch.load(param_path + '/' + name + '.pth')
            self.image.append(image)
            self.param.append(p.squeeze(0))

    
    def __getitem__(self, index):
        return self.image[index], self.param[index]
    
    def __len__(self):
        return len(self.image)



if __name__ == '__main__':
    image_root_path = '/data1/gxy_ctl/ffhq/images'
    param_root_path = '/data1/gxy_ctl/ffhq/param'

    batch_size = 20
    epoch_size = 100

    mydataset = MyDataset(image_root_path, param_root_path)
    dataloader = DataLoader(dataset=mydataset, batch_size=10, shuffle=True, drop_last=False)
    print('\nload ok')
    net = Encoder().cuda()
    optimizer = optim.SGD(net.parameters(), lr = 0.001)

    for epoch in range(epoch_size):
        loss_sum = 0
        cnt = 0
        for i, (image, param) in enumerate(dataloader):
            image, param = image.cuda(), param.cuda()
            optimizer.zero_grad()
            loss = net(image, param)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            cnt += 1
            print(f'\repoch = {epoch}, loader = {str(i)} / {str(len(dataloader))}', end='')
        if (epoch + 1) % 20 == 0:
            print(f'\rEpoch: {epoch}\tloss = {str(loss_sum / cnt)}')
    torch.save(net.state_dict(), './pretrain_model/encoder.pth')
            


    
