import argparse
import time
import torch
import torchvision.utils as vutils

from helpers import vlen, ensure_dir
from hvae import HVAE
from fmnist import FMNIST

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

def kl(p1, p2):
    p1 = p1 * (1-1e-6) + torch.ones_like(p1)/2 * 1e-6
    p2 = p2 * (1-1e-6) + torch.ones_like(p2)/2 * 1e-6
    return p1*(torch.log(p1)-torch.log(p2)) + (1-p1)*(torch.log(1-p1)-torch.log(1-p2))

# entry point, the main
def main():
    time0 = time.time()

    # printout
    logname = './logs/log-' + args.call_prefix + '.txt'
    print('# Starting at ' + time.strftime('%c'), file=open(logname, 'w'), flush=True)

    device = torch.cuda.current_device()
    torch.autograd.set_detect_anomaly(True)

    # the model
    model = HVAE(args.nz0, args.nz1, args.stepsize, device).to(device)
    print('# Model prepared', file=open(logname, 'a'), flush=True)

    if args.load_prefix != '':
        loadprot = model.load_state_dict(torch.load('./models/m_' + args.load_prefix + '.pt'), strict=True)
        print('model: ', loadprot, flush=True)
        print('# Model loaded', file=open(logname, 'a'), flush=True)

    fmnist = FMNIST(args.bs)
    print('# Dataset loaded', file=open(logname, 'a'), flush=True)

    log_period = 100
    save_period = 1000
    niterations = args.niterations
    count = 0

    print('# Everything prepared, go ...', file=open(logname, 'a'), flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.stepsize)

    loss_data = torch.zeros([], device=device)
    loss_kl0 = torch.zeros([], device=device)
    loss_kl1 = torch.zeros([], device=device)

    x_sta = torch.rand([10,1,28,28], device=device)
    
    while True:
        # accumulation speed
        afactor = (1/(count + 1)) if count < 1000 else 0.001
        afactor1 = 1 - afactor

        # MNIST
        # get the data
        x_gt0 = fmnist.get_batch(device)
        # add noise
        x_gt = x_gt0 + torch.randn_like(x_gt0)*0.01

        # learn
        optimizer.zero_grad()
        model.train()

        a0, a1 = model.rnet(x_gt)

        scores0, _ = model.fnet0(args.bs)
        probs0 = scores0.sigmoid()
        scores0_p = scores0 + a0
        probs0_p = scores0_p.sigmoid()
        z0 = ZGR_binary(scores0_p)

        scores1, _ = model.fnet01(z0)
        probs1 = scores1.sigmoid()
        scores1_p = scores1 + a1
        probs1_p = scores1_p.sigmoid()
        z1 = ZGR_binary(scores1_p)

        mu = model.fnet1x(z1)
        lsigma = model.lsigma
        sigma = torch.exp(lsigma)
        d_l = 0.91894 + lsigma + ((x_gt - mu)**2) / (2*(sigma**2))
        d_l = d_l.sum([1,2,3]).mean()

        kl0 = kl(probs0_p, probs0).sum(1).mean()
        kl1 = kl(probs1_p, probs1).sum(1).mean()
        loss = d_l + kl0 + kl1

        loss.backward()
        optimizer.step()

        loss_data = loss_data*afactor1 + d_l.detach()/(28*28)*afactor
        loss_kl0 = loss_kl0*afactor1 + (kl0.detach()/args.nz0)*afactor
        loss_kl1 = loss_kl1*afactor1 + (kl1.detach()/args.nz1)*afactor

        model.update_stats_p(args.bs)
        model.update_stats_q(x_gt)

        x_sta = model.limiting(x_sta)[0]

        # once awhile print something out
        if count % log_period == log_period-1:
            strtoprint = 'time: ' + str(time.time()-time0) + ' count: ' + str(count)

            strtoprint += ' len: ' + str(vlen(model).cpu().numpy())
            strtoprint += ' dloss: ' + str(loss_data.cpu().numpy())
            strtoprint += ' kl0loss: ' + str(loss_kl0.cpu().numpy())
            strtoprint += ' kl1loss: ' + str(loss_kl1.cpu().numpy())

            strtoprint += ' pmi1: ' + str(model.z1_p_mi.mean().cpu().numpy())
            strtoprint += ' qmi0: ' + str(model.z0_q_mi.mean().cpu().numpy())
            strtoprint += ' qmi1: ' + str(model.z1_q_mi.mean().cpu().numpy())

            print(strtoprint, file=open(logname, 'a'), flush=True)

        # once awhile save the models for further use
        if count % save_period == 0:
            print('# Saving models ...', file=open(logname, 'a'), flush=True)
            torch.save(model.state_dict(), './models/m_' + args.call_prefix + '.pt')

            # image
            viz1 = x_gt0[0:10]
            viz2 = model.limiting(x_gt[0:10])[1]
            viz3 = model.single_shot(10)
            viz4, viz5 = model.limiting(x_sta)

            imtoviz = torch.cat((viz1, viz2, viz3, viz4, viz5), dim=0)
            vutils.save_image(imtoviz, './images/img_' + args.call_prefix + '.png', nrow=10)

            # saving stats for pz1
            p1mutin = model.z1_p_mi.cpu().numpy()
            p1mutin.sort()
            with open('./logs/statp1-' + args.call_prefix + '.txt', 'w') as f:
                for i in range(args.nz1):
                    print(p1mutin[i], file=f, flush=True)

            # saving stats for qz0
            q0mutin = model.z0_q_mi.cpu().numpy()
            q0mutin.sort()
            with open('./logs/statq0-' + args.call_prefix + '.txt', 'w') as f:
                for i in range(args.nz0):
                    print(q0mutin[i], file=f, flush=True)

            # saving stats for qz1
            q1mutin = model.z1_q_mi.cpu().numpy()
            q1mutin.sort()
            with open('./logs/statq1-' + args.call_prefix + '.txt', 'w') as f:
                for i in range(args.nz1):
                    print(q1mutin[i], file=f, flush=True)

            print('# ... done.', file=open(logname, 'a'), flush=True)

        count += 1
        if count == niterations:
            break

    print('# Finished at ' + time.strftime('%c') + ', %g seconds elapsed' %
          (time.time()-time0), file=open(logname, 'a'), flush=True)


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

    main()
