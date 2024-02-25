import argparse
import time
import torch
import torchvision.utils as vutils

from helpers import vlen, ensure_dir
from hvae import HVAE
from mnist import MNIST

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--call_prefix', default='tmp', help='Call prefix.')
    parser.add_argument('--load_prefix', default='', help='Load prefix.')
    parser.add_argument('--stepsize', type=float, default=1e-8, help='Gradient step size.')
    parser.add_argument('--bs', type=int, default=1024, help='Batch size')
    parser.add_argument('--niterations', type=int, default=-1, help='')
    parser.add_argument('--nz0', type=int, default=-1, help='')
    parser.add_argument('--nz1', type=int, default=-1, help='')

    args = parser.parse_args()

    ensure_dir('./logs')
    ensure_dir('./models')
    ensure_dir('./images')

    time0 = time.time()

    # printout
    logname = './logs/log-' + args.call_prefix + '.txt'
    print('# Starting at ' + time.strftime('%c'), file=open(logname, 'w'), flush=True)

    device = torch.cuda.current_device()
    torch.autograd.set_detect_anomaly(True)

    # the model
    model = HVAE(args.nz0, args.nz1, args.stepsize).to(device)
    print('# Model prepared', file=open(logname, 'a'), flush=True)

    if args.load_prefix != '':
        loadprot = model.load_state_dict(torch.load('./models/m_' + args.load_prefix + '.pt'), strict=True)
        print('model: ', loadprot, flush=True)
        print('# Model loaded', file=open(logname, 'a'), flush=True)

    mnist = MNIST(args.bs)
    print('# Dataset loaded', file=open(logname, 'a'), flush=True)

    log_period = 100
    save_period = 1000
    niterations = args.niterations
    count = 0

    print('# Everything prepared, go ...', file=open(logname, 'a'), flush=True)

    loss_p = torch.zeros([], device=device)
    loss_q = torch.zeros([], device=device)

    x_sta = (torch.rand([10,1,28,28], device=device)<0.5).float()
    
    while True:
        # accumulation speed
        afactor = (1/(count + 1)) if count < 1000 else 0.001
        afactor1 = 1 - afactor

        # get the data
        x_gt0 = mnist.get_batch(device)
        # add noise to MNIST
        mask = (torch.rand_like(x_gt0)<0.001).float()
        x_gt = x_gt0*(1-mask) + (1-x_gt0)*mask

        # learn
        z0, z1 = model.sample_q(x_gt)
        loss_p = loss_p * afactor1 + model.optimize_p(z0, z1, x_gt) * afactor

        z0, z1, x = model.sample_p(args.bs)
        loss_q = loss_q * afactor1 + model.optimize_q(z0, z1, x) * afactor

        # update limiting sample
        x_sta = model.limiting(x_sta).bernoulli()

        # once awhile print something out
        if count % log_period == log_period-1:
            strtoprint = 'time: ' + str(time.time()-time0) + ' count: ' + str(count)

            strtoprint += ' len: ' + str(vlen(model).cpu().numpy())
            strtoprint += ' ploss: ' + str(loss_p.cpu().numpy())
            strtoprint += ' qloss: ' + str(loss_q.cpu().numpy())

            print(strtoprint, file=open(logname, 'a'), flush=True)

        # once awhile save the models for further use
        if count % save_period == 0:
            print('# Saving models ...', file=open(logname, 'a'), flush=True)
            torch.save(model.state_dict(), './models/m_' + args.call_prefix + '.pt')

            # image
            viz1 = x_gt0[0:10]
            viz2 = model.limiting(x_gt[0:10])
            viz3 = model.single_shot(10)            
            viz4 = model.limiting(x_sta)

            imtoviz = torch.cat((viz1, viz2, viz3, viz4), dim=0)
            vutils.save_image(imtoviz, './images/img_' + args.call_prefix + '.png', nrow=10)

            print('# ... done.', file=open(logname, 'a'), flush=True)

        count += 1
        if count == niterations:
            break

    print('# Finished at ' + time.strftime('%c') + ', %g seconds elapsed' %
          (time.time()-time0), file=open(logname, 'a'), flush=True)
