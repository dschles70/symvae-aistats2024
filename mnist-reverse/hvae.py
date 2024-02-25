import torch

# q(z1|x)
class QNETx1(torch.nn.Module):
    def __init__(self,
                 nz1 : int,
                 nn : int = 32):
        
        super(QNETx1, self).__init__()

        self.actf = torch.tanh

        nnmlp = 600

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

    def forward(self,
                x : torch.Tensor) -> torch.Tensor:

        hh = self.actf(self.conv0(x))
        hh = self.actf(self.conv1(hh))
        hh = self.actf(self.conv2(hh))
        hh = self.actf(self.conv3(hh))
        hh = self.actf(self.conv4(hh))
        hh = self.actf(self.conv5(hh))
        hh = self.actf(self.conv6(hh).squeeze(-1).squeeze(-1))

        return self.conv_z1(hh)

# q(z0|z1)
class QNET10(torch.nn.Module):
    def __init__(self,
                 nz0 : int,
                 nz1 : int):
        
        super(QNET10, self).__init__()

        self.actf = torch.tanh

        nnmlp = 600

        self.conv_z1 = torch.nn.Linear(nz1, nnmlp, bias=True)

        self.lin1 = torch.nn.Linear(nnmlp, nnmlp, bias=True)
        self.lin2 = torch.nn.Linear(nnmlp, nnmlp, bias=True)
        self.lin3 = torch.nn.Linear(nnmlp, nz0,   bias=True)
        
        self.lin_skip = torch.nn.Linear(nz1, nz0, bias=False)

    def forward(self,
                z1 : torch.Tensor) -> torch.Tensor:

        hh = self.actf(self.conv_z1(z1))

        hh = self.actf(self.lin1(hh))
        hh = self.actf(self.lin2(hh))
        return self.lin3(hh) + self.lin_skip(z1)

# p(z1|z0)
class PNET01(torch.nn.Module):
    def __init__(self,
                 nz0 : int,
                 nz1 : int):
        
        super(PNET01, self).__init__()

        self.actf = torch.tanh

        nnmlp = 600

        self.lin1 = torch.nn.Linear(nz0,   nnmlp, bias=True)
        self.lin2 = torch.nn.Linear(nnmlp, nnmlp, bias=True)
        self.lin3 = torch.nn.Linear(nnmlp, nz1,   bias=True)

        self.lin_skip = torch.nn.Linear(nz0, nz1, bias=False)

    def forward(self,
                z0 : torch.Tensor) -> torch.Tensor:
        
        hh = self.actf(self.lin1(z0))
        hh = self.actf(self.lin2(hh))
        return self.lin3(hh) + self.lin_skip(z0)

# p(x|z1)
class PNET1x(torch.nn.Module):
    def __init__(self,
                 nz1 : int,
                 nn : int = 32):
        
        super(PNET1x, self).__init__()

        self.actf = torch.tanh

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
        scores_x = self.conv7(hh)

        return scores_x

class HVAE(torch.nn.Module):
    def __init__(self,
                 nz0 : int,
                 nz1 : int,
                 step : float):
        
        super(HVAE, self).__init__()

        self.nz0 = nz0

        self.pnet01 = PNET01(nz0, nz1)
        self.pnet1x = PNET1x(nz1)
        self.qnetx1 = QNETx1(nz1)
        self.qnet10 = QNET10(nz0, nz1)

        self.optimizer_p = torch.optim.Adam([
            {'params': self.pnet01.parameters()},
            {'params': self.pnet1x.parameters()}
        ], lr=step)

        self.optimizer_q = torch.optim.Adam([
            {'params': self.qnet10.parameters()},
            {'params': self.qnetx1.parameters()}
        ], lr=step)

        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

        self.dummy_param = torch.nn.Parameter(torch.zeros([]))

    # sampling from p
    def sample_p(self,
                 bs : int) -> tuple:
        
        with torch.no_grad():
            z0 = (torch.rand([bs, self.nz0], device = self.dummy_param.device)>0.5).float()
            z1 = self.pnet01(z0).sigmoid().bernoulli().detach()
            x = self.pnet1x(z1).sigmoid().bernoulli().detach()
            return z0, z1, x

    # sampling from q
    def sample_q(self,
                 x : torch.Tensor) -> tuple:
        
        with torch.no_grad():
            z1 = self.qnetx1(x).sigmoid().bernoulli().detach()
            z0 = self.qnet10(z1).sigmoid().bernoulli().detach()
            return z0, z1

    def optimize_p(self, 
                   z0_gt : torch.Tensor,
                   z1_gt : torch.Tensor,
                   x_gt : torch.Tensor) -> torch.Tensor:
        
        self.optimizer_p.zero_grad()

        z1_scores = self.pnet01(z0_gt)
        loss_z1 = self.criterion(z1_scores, z1_gt).sum(1).mean()

        x_scores = self.pnet1x(z1_gt)
        loss_x = self.criterion(x_scores, x_gt).sum([1,2,3]).mean()

        loss = loss_z1 + loss_x
        loss.backward()
        self.optimizer_p.step()
        return loss.detach()/(28*28 + z1_gt.shape[1])

    def optimize_q(self,
                   z0_gt : torch.Tensor,
                   z1_gt : torch.Tensor,
                   x_gt : torch.Tensor):
        
        self.optimizer_q.zero_grad()

        z1_scores = self.qnetx1(x_gt)
        loss_z1 = self.criterion(z1_scores, z1_gt).sum(1).mean()

        z0_scores = self.qnet10(z1_gt)
        loss_z0 = self.criterion(z0_scores, z0_gt).sum(1).mean()

        loss = loss_z0 + loss_z1
        loss.backward()
        self.optimizer_q.step()
        return loss.detach()/(z0_gt.shape[1] + z1_gt.shape[1])

    def single_shot(self,
                    bs : int) -> torch.Tensor:
        
        with torch.no_grad():
            z0 = (torch.rand([bs, self.nz0], device = self.dummy_param.device)>0.5).float()
            z1 = self.pnet01(z0).sigmoid().bernoulli().detach()
            return self.pnet1x(z1).sigmoid().detach()
        
    def limiting(self,
                 x : torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            z1 = self.qnetx1(x).sigmoid().bernoulli().detach()
            return self.pnet1x(z1).sigmoid().detach()
