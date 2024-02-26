import argparse
import numpy as np
import torch
import torchvision.utils as vutils

from hvae import HVAE

from helpers import ensure_dir
from mnist import MNIST

# entry point, the main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--call_prefix', required=True, help='Call prefix.')
    parser.add_argument('--mode', required=True, help='Call prefix.')
    parser.add_argument('--nz0', type=int, required=True, help='')
    parser.add_argument('--nz1', type=int, required=True, help='')

    args = parser.parse_args()

    # printout
    print('# Start ...', flush=True)

    device = torch.cuda.current_device()
    torch.autograd.set_detect_anomaly(True)

    bs = 1000 # batch size
    nz0 = args.nz0
    nz1 = args.nz1
    call_prefix = args.call_prefix
    mode = args.mode

    # the model
    model = HVAE(nz0, nz1, 0).to(device)

    loadprot = model.load_state_dict(torch.load('./models/m_' + call_prefix + '.pt'), strict=True)
    print('# Model: ', loadprot, flush=True)

    mnist = MNIST(bs)
    print('# Dataset loaded,', mnist.n, 'samples in total.', flush=True)

    # prepare folders
    ensure_dir('./generated_data/' + mode + '/images')
    ensure_dir('./generated_data/' + mode + '/sh_images')
    ensure_dir('./generated_data/' + mode + '/lim_images')
    
    xall = torch.split(mnist.x, bs) # data batches

    for b in range(len(xall)):
        # current batch
        xcurr = xall[b].to(device)

        # save original images
        for i in range(bs):
            vutils.save_image(xcurr[i], './generated_data/' + mode + '/images/img_%05d.png' % (b*bs+i))

        # single-shot images
        x = (model.single_shot(bs)>0.5).float()
        for i in range(bs):
            vutils.save_image(x[i], './generated_data/' + mode + '/sh_images/img_%05d.png' % (b*bs+i))

        # limiting
        x = (model.single_shot(bs)>0.5).float()
        for _ in range(1000):
            x = model.limiting(x).bernoulli()
        x = (model.limiting(x)>0.5).float()

        for i in range(bs):
            vutils.save_image(x[i], './generated_data/' + mode + '/lim_images/img_%05d.png' % (b*bs+i))

        print('.', flush=True, end='')

    print(' done.')
