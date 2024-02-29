import argparse
import torch
import torchvision.utils as vutils

from hvae import HVAE

from helpers import ensure_dir
from fmnist import FMNIST

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

    # nz0 = 50
    # nz1 = 200

    # call_prefix = '21' # ELBO
    # mode = 'elbo'

    # # call_prefix = '11' # symmetric
    # # mode = 'sym'

    # the model
    model = HVAE(nz0, nz1, 0, device).to(device)

    loadprot = model.load_state_dict(torch.load('./models/m_' + call_prefix + '.pt'), strict=True)
    print('# Model: ', loadprot, flush=True)

    fmnist = FMNIST(bs)
    print('# Dataset loaded,', fmnist.n, 'samples in total.', flush=True)

    # prepare folders
    ensure_dir('./generated_data/' + mode + '/images')
    ensure_dir('./generated_data/' + mode + '/sh_images')
    ensure_dir('./generated_data/' + mode + '/lim_images')
    
    xall = torch.split(fmnist.x, bs) # data batches

    for b in range(len(xall)):
        # current batch
        xcurr = xall[b].to(device)

        # save original images
        for i in range(bs):
            vutils.save_image(xcurr[i], './generated_data/' + mode + '/images/img_%05d.png' % (b*bs+i))

        # prior z, single-shot images
        x = model.single_shot(bs)
        for i in range(bs):
            vutils.save_image(x[i], './generated_data/' + mode + '/sh_images/img_%05d.png' % (b*bs+i))

        # limiting
        x = (torch.rand([bs,1,28,28], device=device)<0.5).float()
        for _ in range(1000):
            x = model.limiting(x)[0]
        x = model.limiting(x)[1]
            
        for i in range(bs):
            vutils.save_image(x[i], './generated_data/' + mode + '/lim_images/img_%05d.png' % (b*bs+i))

        print('.', flush=True, end='')

    print(' done.')
