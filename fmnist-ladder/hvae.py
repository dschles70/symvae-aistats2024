import torch

# this network is deterministic, provides "additional" scores for posterior q
class RNET(torch.nn.Module):
    def __init__(self,
                 nz0 : int,
                 nz1 : int,
                 nn = 32):
        super(RNET, self).__init__()

        self.actf = torch.tanh

        nnmlp = 1000

        self.conv0 = torch.nn.Conv2d(1,   nn,  3, 1, padding=1, padding_mode='zeros', bias=True)

        # 3*28*28 ->
        self.conv1 = torch.nn.Conv2d(nn,      nn,      2, 1, bias=True) # 27
        self.conv2 = torch.nn.Conv2d(nn,      nn * 2,  3, 2, bias=True) # 13
        self.conv3 = torch.nn.Conv2d(nn * 2,  nn * 2,  3, 1, bias=True) # 11
        self.conv4 = torch.nn.Conv2d(nn * 2,  nn * 2,  3, 1, bias=True) # 9
        self.conv5 = torch.nn.Conv2d(nn * 2,  nn * 4,  3, 2, bias=True) # 4
        self.conv6 = torch.nn.Conv2d(nn * 4,  nnmlp,     4, 1, bias=True) # 1
        # z1 here
        self.conv_z1 = torch.nn.Linear(nnmlp, nz1, bias=True)

        self.lin1 = torch.nn.Linear(nnmlp, nnmlp, bias=True)
        self.lin2 = torch.nn.Linear(nnmlp, nnmlp, bias=True)
        self.lin3 = torch.nn.Linear(nnmlp, nz0,   bias=True)
        
        self.lin_skip = torch.nn.Linear(nz1, nz0, bias=False)

    def forward(self,
                x : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        hh = self.actf(self.conv0(x))
        hh = self.actf(self.conv1(hh))
        hh = self.actf(self.conv2(hh))
        hh = self.actf(self.conv3(hh))
        hh = self.actf(self.conv4(hh))
        hh = self.actf(self.conv5(hh))
        hh = self.actf(self.conv6(hh).squeeze(-1).squeeze(-1))

        s1 = self.conv_z1(hh)

        hh = self.actf(self.lin1(hh))
        hh = self.actf(self.lin2(hh))
        s0 = self.lin3(hh) + self.lin_skip(s1)

        return s0, s1

# p(z0)/p(z0|x)
class FNET0(torch.nn.Module):
    def __init__(self,
                 nz0 : int,
                 device : int):
        super(FNET0, self).__init__()

        self.nz0 = nz0
        self.device = device

    def forward(self,
                bs : int,
                s0 : torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        
        scores_z0 = torch.zeros([bs,self.nz0], device=self.device)
        if s0 != None:
            scores_z0 = scores_z0 + s0

        z0 = scores_z0.sigmoid().bernoulli().detach()

        return scores_z0, z0

# p(z1|z0)/p(z1|z0,x)
class FNET01(torch.nn.Module):
    def __init__(self,
                 nz0 : int,
                 nz1 : int):
        super(FNET01, self).__init__()

        self.actf = torch.tanh

        nnmlp = 600

        self.lin1 = torch.nn.Linear(nz0,   nnmlp, bias=True)
        self.lin2 = torch.nn.Linear(nnmlp, nnmlp, bias=True)
        self.lin3 = torch.nn.Linear(nnmlp, nz1,   bias=True)

        self.lin_skip = torch.nn.Linear(nz0, nz1, bias=False)

    def forward(self,
                z0 : torch.Tensor,
                s1 : torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        
        hh = self.actf(self.lin1(z0))
        hh = self.actf(self.lin2(hh))
        scores_z1 = self.lin3(hh) + self.lin_skip(z0)

        if s1 != None:
            scores_z1 = scores_z1 + s1

        z1 = scores_z1.sigmoid().bernoulli().detach()

        return scores_z1, z1

# p(x|z1)
class FNET1x(torch.nn.Module):
    def __init__(self,
                 nz1 : int,
                 nn : int = 32):
        super(FNET1x, self).__init__()

        self.actf = torch.relu

        self.conv1 = torch.nn.ConvTranspose2d(nz1,    nn * 4, 4, 1, bias=True) # 4
        self.conv2 = torch.nn.ConvTranspose2d(nn * 4, nn * 2, 3, 2, bias=True) # 9
        self.conv3 = torch.nn.ConvTranspose2d(nn * 2, nn * 2, 3, 1, bias=True) # 11
        self.conv4 = torch.nn.ConvTranspose2d(nn * 2, nn * 2, 3, 1, bias=True) # 13
        self.conv5 = torch.nn.ConvTranspose2d(nn * 2, nn,     3, 2, bias=True) # 27
        self.conv6 = torch.nn.ConvTranspose2d(nn,     nn,     2, 1, bias=True) # 28
        self.conv7 = torch.nn.Conv2d(nn, 1, 3, 1, padding=1, padding_mode='zeros', bias=True) # 28

    def forward(self,
                z1 : torch.Tensor) -> torch.Tensor:

        hh = z1.unsqueeze(-1).unsqueeze(-1)
        
        hh = self.actf(self.conv1(hh))
        hh = self.actf(self.conv2(hh))
        hh = self.actf(self.conv3(hh))
        hh = self.actf(self.conv4(hh))
        hh = self.actf(self.conv5(hh))
        hh = self.actf(self.conv6(hh))
        return self.conv7(hh)

class HVAE(torch.nn.Module):
    def __init__(self,
                 nz0 : int,
                 nz1 : int,
                 step : float,
                 device : int):
        super(HVAE, self).__init__()

        self.fnet0 = FNET0(nz0, device)
        self.fnet01 = FNET01(nz0, nz1)
        self.fnet1x = FNET1x(nz1)
        self.rnet = RNET(nz0, nz1)

        self.lsigma = torch.nn.Parameter(torch.zeros([]), requires_grad=True)

        self.optimizer_p = torch.optim.Adam([
            {'params': self.fnet01.parameters()},
            {'params': self.fnet1x.parameters()},
            {'params': self.lsigma}
        ], lr=step)

        self.optimizer_q = torch.optim.Adam(self.rnet.parameters(), lr=step)

        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

        self.z0_q = torch.nn.Parameter(torch.ones([nz0])/2, requires_grad=False)
        self.z0_q_mi = torch.nn.Parameter(torch.zeros([nz0]), requires_grad=False)

        self.z1_p = torch.nn.Parameter(torch.ones([nz1])/2, requires_grad=False)
        self.z1_p_mi = torch.nn.Parameter(torch.zeros([nz1]), requires_grad=False)

        self.z1_q = torch.nn.Parameter(torch.ones([nz1])/2, requires_grad=False)
        self.z1_q_mi = torch.nn.Parameter(torch.zeros([nz1]), requires_grad=False)

    # sampling from p
    def sample_p(self,
                 bs : int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        with torch.no_grad():
            _, z0 = self.fnet0(bs)
            _, z1 = self.fnet01(z0)
            mu = self.fnet1x(z1)
            x = torch.randn_like(mu) * torch.exp(self.lsigma) + mu
            return z0, z1, x

    # sampling from q
    def sample_q(self,
                 x : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        with torch.no_grad():
            a0, a1 = self.rnet(x)
            bs = x.shape[0]
            _, z0 = self.fnet0(bs, s0 = a0)
            _, z1 = self.fnet01(z0, s1 = a1)
            return z0, z1

    def optimize_p(self,
                   z0_gt : torch.Tensor,
                   z1_gt : torch.Tensor,
                   x_gt : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        self.optimizer_p.zero_grad()

        z1_scores, _ = self.fnet01(z0_gt)
        loss_z1 = self.criterion(z1_scores, z1_gt).sum(1).mean()

        mu = self.fnet1x(z1_gt)
        sigma = torch.exp(self.lsigma)
        loss_x = 0.91894 + self.lsigma + ((x_gt - mu)**2) / (2*(sigma**2))
        loss_x = loss_x.sum([1,2,3]).mean()

        loss = loss_z1 + loss_x
        loss.backward()
        self.optimizer_p.step()

        return loss_z1.detach()/z1_gt.shape[1], loss_x.detach()/(x_gt.shape[1]*x_gt.shape[2]*x_gt.shape[3])

    def optimize_q(self,
                   z0_gt : torch.Tensor,
                   z1_gt : torch.Tensor,
                   x_gt : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        self.optimizer_q.zero_grad()

        a0, a1 = self.rnet(x_gt)
        bs = x_gt.shape[0]

        z0_scores, _ = self.fnet0(bs, s0 = a0)
        loss_z0 = self.criterion(z0_scores, z0_gt).sum(1).mean()

        z1_scores, _ = self.fnet01(z0_gt)
        z1_scores = z1_scores.detach() + a1
        loss_z1 = self.criterion(z1_scores, z1_gt).sum(1).mean()

        loss = loss_z0 + loss_z1
        loss.backward()
        self.optimizer_q.step()

        return loss_z0.detach()/z0_gt.shape[1], loss_z1.detach()/z1_gt.shape[1]

    def update_stats_p(self,
                       bs : int):
        
        with torch.no_grad():
            _, z0 = self.fnet0(bs)
            scores, _ = self.fnet01(z0)
            probs = scores.sigmoid()
            self.z1_p.data.copy_(self.z1_p.data*0.99 + probs.mean(0)*0.01)

            # weaken probabilities
            p1 = self.z1_p.data*(1-1e-6) + torch.ones_like(self.z1_p.data)/2*1e-6
            p2 = probs*(1-1e-6) + torch.ones_like(probs)/2*1e-6

            h1 = p1*torch.log(p1) + (1-p1)*torch.log(1-p1)
            h2 = p2*torch.log(p2) + (1-p2)*torch.log(1-p2)
            mi = h2.mean(0) - h1

            self.z1_p_mi.data.copy_(self.z1_p_mi.data*0.99 + mi*0.01)

    def update_stats_q(self,
                       x : torch.Tensor):
        
        with torch.no_grad():
            a0, a1 = self.rnet(x)
            bs = x.shape[0]
            scores0, z0 = self.fnet0(bs, s0 = a0)
            scores1, _ = self.fnet01(z0, s1 = a1)

            probs0 = scores0.sigmoid()
            self.z0_q.data.copy_(self.z0_q.data*0.99 + probs0.mean(0)*0.01)

            probs1 = scores1.sigmoid()
            self.z1_q.data.copy_(self.z1_q.data*0.99 + probs1.mean(0)*0.01)

            # weaken probabilities
            p1 = self.z0_q.data*(1-1e-6) + torch.ones_like(self.z0_q.data)/2*1e-6
            p2 = probs0*(1-1e-6) + torch.ones_like(probs0)/2*1e-6

            h1 = p1*torch.log(p1) + (1-p1)*torch.log(1-p1)
            h2 = p2*torch.log(p2) + (1-p2)*torch.log(1-p2)
            mi = h2.mean(0) - h1

            self.z0_q_mi.data.copy_(self.z0_q_mi.data*0.99 + mi*0.01)

            # weaken probabilities
            p1 = self.z1_q.data*(1-1e-6) + torch.ones_like(self.z1_q.data)/2*1e-6
            p2 = probs1*(1-1e-6) + torch.ones_like(probs1)/2*1e-6

            h1 = p1*torch.log(p1) + (1-p1)*torch.log(1-p1)
            h2 = p2*torch.log(p2) + (1-p2)*torch.log(1-p2)
            mi = h2.mean(0) - h1

            self.z1_q_mi.data.copy_(self.z1_q_mi.data*0.99 + mi*0.01)

    # for single-shot
    def single_shot(self,
                    bs : int) -> torch.Tensor:
        
        with torch.no_grad():
            _, z0 = self.fnet0(bs)
            _, z1 = self.fnet01(z0)
            return self.fnet1x(z1)

    # for limiting
    def limiting(self,
                 x : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        with torch.no_grad():
            a0, a1 = self.rnet(x)
            bs = x.shape[0]
            _, z0 = self.fnet0(bs, s0 = a0)
            _, z1 = self.fnet01(z0, s1 = a1)
            mu = self.fnet1x(z1)
            x = torch.randn_like(mu) * torch.exp(self.lsigma) + mu            
            return x, mu
