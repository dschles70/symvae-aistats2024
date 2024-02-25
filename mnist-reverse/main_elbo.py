import argparse
import time
import torch
import torchvision.utils as vutils

from helpers import vlen, ensure_dir
from hvae import HVAE
from mnist import MNIST

def ZGR_binary(logits:torch.Tensor, x:torch.Tensor=None)->torch.Tensor:
    """Returns a Bernoulli sample for given logits with ZGR = DARN(1/2) gradient
    Input: logits [*]
    x: (optional) binary sample to use instead of drawing a new sample. [*]
    Output: binary samples with ZGR gradient [*], dtype as logits
    """
    p = torch.sigmoid(logits)
    if x is None:
        x = p.bernoulli()
    J = (x * (1-p) + (1-x)*p )/2
    return x + J.detach()*(logits - logits.detach()) # value of x with J on backprop to logits

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

    print('# Models prepared, go ...', file=open(logname, 'a'), flush=True)

    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.stepsize)

    loss_data = torch.zeros([], device=device)
    loss_kl = torch.zeros([], device=device)

    x_sta = (torch.rand([10,1,28,28], device=device)<0.5).float()
    
    while True:
        # accumulation speed
        afactor = (1/(count + 1)) if count < 1000 else 0.001
        afactor1 = 1 - afactor

        # MNIST
        # get the data
        x_gt0 = mnist.get_batch(device)
        # add noise to MNIST
        mask = (torch.rand_like(x_gt0)<0.001).float()
        x_gt = x_gt0*(1-mask) + (1-x_gt0)*mask

        # learn
        optimizer.zero_grad()

        scores1 = model.qnetx1(x_gt)
        z1 = ZGR_binary(scores1)
        scores0 = model.qnet10(z1)
        z0 = ZGR_binary(scores0)

        scores1_p = model.pnet01(z0)
        x_scores = model.pnet1x(z1)

        d_l = criterion(x_scores, x_gt).sum([1,2,3]).mean()
        kla = - criterion(scores1, z1).sum(1).mean()
        klb = - criterion(scores0, z0).sum(1).mean()
        klc = 0.6931 * args.nz0
        kld = criterion(scores1_p, z1).sum(1).mean()
        kl = kla + klb + klc + kld

        loss = d_l + kl

        loss.backward()
        optimizer.step()

        loss_data = loss_data*afactor1 + d_l.detach()/(28*28)*afactor
        loss_kl = loss_kl*afactor1 + (kl.detach()/(args.nz0+args.nz1))*afactor

        x_sta = model.limiting(x_sta).bernoulli()

        # once awhile print something out
        if count % log_period == log_period-1:
            strtoprint = 'time: ' + str(time.time()-time0) + ' count: ' + str(count)

            strtoprint += ' len: ' + str(vlen(model).cpu().numpy())
            strtoprint += ' dloss: ' + str(loss_data.cpu().numpy())
            strtoprint += ' klloss: ' + str(loss_kl.cpu().numpy())

            print(strtoprint, file=open(logname, 'a'), flush=True)

        # once awhile save the models for further use
        if count % save_period == 0:
            print('# Saving models ...', file=open(logname, 'a'), flush=True)
            torch.save(model.state_dict(), './models/m_' + args.call_prefix + '.pt')

            # image
            viz1 = x_gt0[0:10]
            viz2 = model.limiting(x_gt0[0:10])
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
