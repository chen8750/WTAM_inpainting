def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'MuFA_Net':
        assert (opt.dataset_mode == 'aligned' or opt.dataset_mode == 'aligned_resized')
        from models.net_models.MuFANet_model import MuFANetModel
        model = MuFANetModel()

    elif opt.model == 'WTAM':
        assert (opt.dataset_mode == 'aligned' or opt.dataset_mode == 'aligned_resized')
        from models.net_models.WTAM_model import WTAMModel
        model = WTAMModel()

    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
