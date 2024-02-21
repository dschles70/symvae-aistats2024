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
    model = HVAE(nz0, nz1, 0, device).to(device)

    loadprot = model.load_state_dict(torch.load('./models/m_' + call_prefix + '.pt'), strict=True)
    print('# Model: ', loadprot, flush=True)

    mnist = MNIST(bs)
    print('# Dataset loaded,', mnist.n, 'samples in total.', flush=True)

    # prepare folders
    ensure_dir('./generated_data/' + mode + '/images')
    ensure_dir('./generated_data/' + mode + '/sh_images')
    ensure_dir('./generated_data/' + mode + '/lim_images')
    
    xall = torch.split(mnist.x, bs) # data batches

    # z-s are stored for tSNE-embeddings, which are not included in this code
    z0_prior = []
    z1_prior = []
    z0_post = []
    z1_post = []
    z0_lim = []
    z1_lim = []

    for b in range(len(xall)):
        # current batch
        xcurr = xall[b].to(device)

        # save original images
        for i in range(bs):
            vutils.save_image(xcurr[i], './generated_data/' + mode + '/images/img_%05d.png' % (b*bs+i))

        # prior z, single-shot images
        z0, z1, _ = model.sample_p(bs)
        z0_prior += [z0]
        z1_prior += [z1]
        with torch.no_grad():
            x = (model.fnet1x(z1)[0]>0).float()
        for i in range(bs):
            vutils.save_image(x[i], './generated_data/' + mode + '/sh_images/img_%05d.png' % (b*bs+i))

        # posterior z
        z0, z1 = model.sample_q(xcurr)
        z0_post += [z0]
        z1_post += [z1]

        # limiting
        _, _, x = model.sample_p(bs)
        for j in range(1000):
            x = model.limiting(x).bernoulli()    
        z0, z1 = model.sample_q(x)
        z0_lim += [z0]
        z1_lim += [z1]
        with torch.no_grad():
            x = (model.fnet1x(z1)[0]>0).float()
        for i in range(bs):
            vutils.save_image(x[i], './generated_data/' + mode + '/lim_images/img_%05d.png' % (b*bs+i))

        print('.', flush=True, end='')

    z0_prior = torch.cat(z0_prior).cpu().numpy()
    np.save('./generated_data/' + mode + '/z0_prior', z0_prior)
    z1_prior = torch.cat(z1_prior).cpu().numpy()
    np.save('./generated_data/' + mode + '/z1_prior', z1_prior)

    z0_post = torch.cat(z0_post).cpu().numpy()
    np.save('./generated_data/' + mode + '/z0_post', z0_post)
    z1_post = torch.cat(z1_post).cpu().numpy()
    np.save('./generated_data/' + mode + '/z1_post', z1_post)

    z0_lim = torch.cat(z0_lim).cpu().numpy()
    np.save('./generated_data/' + mode + '/z0_lim', z0_lim)
    z1_lim = torch.cat(z1_lim).cpu().numpy()
    np.save('./generated_data/' + mode + '/z1_lim', z1_lim)

    print(' done.')
