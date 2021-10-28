class cfg:
    size = 1024
    resize = 256
    mid_size = 64
    print_epoch = 300

    class rec:
        size = 1024
        step = 2000
        lr = 0.05
        w_epochs = 1200
        n_epochs = 400
        fs_epochs = 600
        noise = 0.05
        noise_ramp = 0.75
        print_epoch= 400

    class I2SLoss:
        lamb_mse = 0
        lamb_p = 1e-5
        lamb_noisemse = (1e-5,)

    class stage1:
        lr = 0.01
        epochs = 300
        lamb_mseloss_mid = 0.01
        lamb_lpipsfaceloss_1024 = 0.1
        lamb_lpipsloss = 0.1
        lamb_styleloss = 15000
        lamb_segmentloss = 0.01
        epoch_s1 = 200
        epoch_s2 = 400
    
    class stage2:
        lr = 0.02
        epochs = 300
        
        lamb_styleloss = 20
        lamb_shapeloss = 0.00001
        lamb_segmentloss = 0.0001
        lamb_mseloss_32 = 0.0003
        lamb_mseloss_1024 = 0.001
        lamb_orientloss = 0.0001
        lamb_lpipsloss = 1
    
    class stage3:
        epochs = 600
        lr = 0.02
        lamb_mseloss_1024 = 0.01

    class stage4:
        epochs = 1200
        lr = 0.02
        lamb_mseloss_mid = 0.01
        lamb_mseloss_1024 = 0.001

        
    class modelpath:
        segment = 'pretrain_model/inference.pth'
        styleGAN2 = 'pretrain_model/stylegan2-ffhq-config-f.pt';
    
    class styleGAN:
        size = 1024
        dimention = 512
        noise = 0.005
        noise_ramp = 0.75
        step = 2000
        