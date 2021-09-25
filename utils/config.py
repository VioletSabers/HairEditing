class cfg:
    size = 1024
    resize = 256
    mid_size = 32


    class rec:
        size = 1024
        step = 2000
        lr = 0.05
        w_epochs = 1000
        n_epochs = 500
        fs_epochs = 500
        noise = 0.05
        noise_ramp = 0.75

    class I2SLoss:
        lamb_mse = 1e-5
        lamb_p = 1e-5
        lamb_noisemse = (1e-5,)

    class stage1:
        lr = 0.01
        epochs = 400
        lamb_mseloss_mid = 0.01
        lamb_lpipsfaceloss_1024 = 0.1
        lamb_lpipsloss = 0.1
        lamb_styleloss = 15000
        lamb_segmentloss = 0.001
        lamb_mseloss_32 = 0.001
    
    class stage2:
        lr = 0.02
        epochs = 400
        
        lamb_styleloss = 20
        lamb_shapeloss = 0.00001
        lamb_segmentloss = 0.0001
        lamb_mseloss_32 = 0.0003
        lamb_mseloss_1024 = 0.01
        lamb_orientloss = 0.0001
        lamb_lpipsloss = 2
    
    class stage3:
        epochs = 400
        lr = 0.03
        lamb_mseloss_32 = 0.01
        lamb_mseloss_1024 = 0.01
        
        

    class modelpath:
        segment = 'pretrain_model/inference.pth'
        styleGAN2 = 'pretrain_model/stylegan2-ffhq-config-f.pt';
    
    class styleGAN:
        size = 1024
        dimention = 512
        noise = 0.005
        noise_ramp = 0.75
        step = 2000
        