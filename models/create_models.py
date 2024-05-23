import torch

def create_model(opt):
    if opt.model == 'INRECGAN':
        from .INRECGAN_model import INRECGANModel, InferenceModel
        if opt.isTrain: # 如果进行训练
            model = INRECGANModel()
        else:
            model = InferenceModel()
    else:
        print('Model name error! Should be INRECGAN!')

    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids) and not opt.fp16:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
