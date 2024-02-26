import argparse
import torch
import torchvision.utils as vutils

from hvae import HVAE
from helpers import ensure_dir

# entry point, the main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--call_prefix', default='tmp', help='Call prefix.')
    parser.add_argument('--nz0', type=int, default=-1, help='')
    parser.add_argument('--nz1', type=int, default=-1, help='')

    args = parser.parse_args()

    ensure_dir('./images')

    # printout
    print('# Start ...', flush=True)

    device = torch.cuda.current_device()
    torch.autograd.set_detect_anomaly(True)

    # the model
    model = HVAE(args.nz0, args.nz1, 0).to(device)

    loadprot = model.load_state_dict(torch.load('./models/m_' + args.call_prefix + '.pt'), strict=True)
    print('# Model: ', loadprot, flush=True)

    x_single_shot = model.single_shot(40)
    vutils.save_image(x_single_shot, './images/generated_sh_' + args.call_prefix + '.png', nrow=10)

    print('# Single-shot saved', flush=True)

    x_sta = (torch.rand([40,1,28,28], device=device)<0.5).float()
    for _ in range(1000):
        x_sta = model.limiting(x_sta).bernoulli()

    x_stationary = model.limiting(x_sta)
    vutils.save_image(x_stationary, './images/generated_sta_' + args.call_prefix + '.png', nrow=10)

    print('# Stationary saved', flush=True)

    print('# Done', flush=True)
