import argparse
import time
import torch
import torchvision.utils as vutils

from helpers import vlen, ensure_dir
from hvae import HVAE
from mnist import MNIST

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

    mnist = MNIST(args.bs)
    print('# Dataset loaded', file=open(logname, 'a'), flush=True)

    log_period = 100
    save_period = 1000
    niterations = args.niterations
    count = 0

    print('# Everything prepared, go ...', file=open(logname, 'a'), flush=True)

    loss_p_1 = torch.zeros([], device=device)
    loss_p_x = torch.zeros([], device=device)
    loss_q_0 = torch.zeros([], device=device)
    loss_q_1 = torch.zeros([], device=device)

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
        z0, z1 = model.sample_q(x_gt)
        lp1, lpx = model.optimize_p(z0, z1, x_gt)
        loss_p_1 = loss_p_1*afactor1 + lp1*afactor
        loss_p_x = loss_p_x*afactor1 + lpx*afactor

        z0, z1, x = model.sample_p(args.bs)
        lq0, lq1 = model.optimize_q(z0, z1, x)
        loss_q_0 = loss_q_0*afactor1 + lq0*afactor
        loss_q_1 = loss_q_1*afactor1 + lq1*afactor

        model.update_stats_p(args.bs)
        model.update_stats_q(x_gt)

        # update limiting sample
        x_sta = model.limiting(x_sta).bernoulli()

        # once awhile print something out
        if count % log_period == log_period-1:
            strtoprint = 'time: ' + str(time.time()-time0) + ' count: ' + str(count)

            strtoprint += ' len: ' + str(vlen(model).cpu().numpy())
            strtoprint += ' ploss1: ' + str(loss_p_1.cpu().numpy())
            strtoprint += ' plossx: ' + str(loss_p_x.cpu().numpy())
            strtoprint += ' qloss0: ' + str(loss_q_0.cpu().numpy())
            strtoprint += ' qloss1: ' + str(loss_q_1.cpu().numpy())

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
            viz2 = model.limiting(x_gt[0:10])
            viz3 = model.single_shot(10)
            viz4 = model.limiting(x_sta)

            imtoviz = torch.cat((viz1, viz2, viz3, viz4), dim=0)
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
