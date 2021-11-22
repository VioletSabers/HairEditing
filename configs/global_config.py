class cfg:
    size = 1024
    
    device = "cuda:0"
    exp_dir = "./results"
    class restyle_e4e:
        size = 256
        input_nc = 6
        n_iters_per_batch = 5

    class modelpath:
        restyle_e4e = "pretrained_model/restyle_e4e_ffhq.pt"
        styleGAN2 = 'pretrained_model/styleGAN2_ffhq.pt'
        latent_avg = "pretrained_model/latent_avg_ffhq.pt"
        segment = "pretrained_model/inference.pt"

    class rec:
        restyle4e = True
        
        lamb_lpipsloss = 1e-5
        lamb_mseloss_1024 = 1e-4
        mid = 9
        lr = 0.01
        lr2 = 0.02
        w_epochs = 100
        style_epochs = 0
        n_epochs = 0
        lamb_lpips = 1e-4
        lamb_mse_hair = 0.0001
        lamb_c = 1e-3
        lamb_styleloss = 10000
        noise = 0.005
        noise_ramp = 0.75
        print_epoch = 100
        step = 2000
    
    class styleGAN:
        size = 1024
        dimention = 512
        noise = 0.005
        noise_ramp = 0.75
        step = 2000
    
    class blend:
        lr = 0.05
        mid = 13
        
        print_epoch = 200
        mid_size = 64
        epoch1 = 400
        lamb1_mseloss_1024_hair = 1e-3
        lamb1_mseloss_1024_face = 1e-4
        lamb1_w = 1e-1
        lamb1_lpipsloss = 1e-5
    
