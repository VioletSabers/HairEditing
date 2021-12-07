import torch
import torch.optim as optim

a = torch.rand([1, 512], requires_grad=True)
print(a.requires_grad)

opt = optim.Adam([a], lr=0.01, betas=(0.9, 0.999), eps=1e-8)

b = 2 * a
loss = b.norm()
loss.backward()
opt.step()

print(a)