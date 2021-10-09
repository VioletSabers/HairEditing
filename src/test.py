import torch

def RGBtoHSV(C):
    R = C[:,0,:,:].unsqueeze(1)
    G = C[:,1,:,:].unsqueeze(1)
    B = C[:,2,:,:].unsqueeze(1)
    max, argmax = torch.max(C, dim=1)

    min, argmin = torch.min(C, dim=1)

    max = max.unsqueeze(1)
    argmax = argmax.unsqueeze(1)
    min = min.unsqueeze(1)
    V, _ = torch.max(C, dim=1)
    V = V.unsqueeze(0)
    S = (max-min)/max
    H = (G-B)/(max-min)* 60 * (argmax == 0).float()
    H = H + (120 + (B-R)/(max-min) * 60) * (argmax == 1).float()
    H = H + (240 + (R-G)/(max-min) * 60) * (argmax == 2).float()
    
    H = (H + 360) % 360
    return torch.cat([H, S, V], dim=1)

def HSVtoRGB(C: torch.Tensor):
    H = C[:,0,:,:].unsqueeze(1)
    S = C[:,1,:,:].unsqueeze(1)
    V = C[:,2,:,:].unsqueeze(1)
    zeros = (S < 1e-4).float()
    angel = (H / 60).int()

    f = H / 60 - angel
    a = V * (1 - S)
    b = V * (1 - S * f)
    c = V * (1 - S * (1 - f))
    
    R = ((angel == 0).float() * V + \
        (angel == 1).float() * b + \
        (angel == 2).float() * a + \
        (angel == 3).float() * a + \
        (angel == 4).float() * c + \
        (angel == 5).float() * V) * (1 - zeros) + V * zeros
    
    G = ((angel == 0).float() * c + \
        (angel == 1).float() * V + \
        (angel == 2).float() * V + \
        (angel == 3).float() * b + \
        (angel == 4).float() * a + \
        (angel == 5).float() * a) * (1 - zeros) + V * zeros
    
    B = ((angel == 0).float() * a + \
        (angel == 1).float() * a + \
        (angel == 2).float() * c + \
        (angel == 3).float() * V + \
        (angel == 4).float() * V + \
        (angel == 5).float() * b) * (1 - zeros) + V * zeros
    
    return torch.cat([R, G, B], dim=1)


C_RGB = torch.rand(1, 3, 4, 4)
C_HSV = RGBtoHSV(C_RGB)
C_RGB1 = HSVtoRGB(C_HSV)

print(C_RGB)
print(C_RGB1)