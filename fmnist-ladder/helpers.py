import os

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def vlen(layer):
    vsum = 0.
    vnn = 0
    for vv in layer.parameters():
        if vv.requires_grad:
            param = vv.data
            vsum = vsum + (param*param).sum()
            vnn = vnn + param.numel()
    return vsum/vnn
