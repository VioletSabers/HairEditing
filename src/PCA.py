import numpy as np
import torch
from sklearn.decomposition import PCA

import sys
sys.path.append("./")
from Base import *
from utils.c_utils import *

data = torch.load('./results/0/latent_all.pth').detach().cpu()
P = []
for i in range(18):
    pca = PCA(n_components=5)
    data_i = data[:, i, :]
    pca.fit(np.array(data_i))
    P.append(pca)

latent = []

for i in range(18):
    new_latent = P[i].transform(data[:, i, :])
    latent.append(torch.from_numpy(P[i].inverse_transform(new_latent)).unsqueeze(1))
    # latent.append(data0[:, i, :])

latent = torch.cat(latent, dim=1)
for i in range(len(latent)):
    with torch.no_grad():
        syn_img, _ = G([latent[i].unsqueeze(0).float().cuda()], input_is_latent=True)
    save_img(syn_img)



