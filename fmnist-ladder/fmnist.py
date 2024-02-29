import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms

class FMNIST():
    def __init__(self,
                 bs : int):

        self.bs = bs

        # load data
        train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform=transforms.Compose([transforms.ToTensor()]))
        test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=transforms.Compose([transforms.ToTensor()])) 

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100)

        x = []
        for images, _ in train_loader:
            x += [images]
        for images, _ in test_loader:
            x += [images]

        self.x = torch.stack(x).flatten(0, 1)

        self.n = self.x.shape[0]
        self.period = self.n//bs
        self.count = 0

    def get_batch(self,
                  device : int) -> torch.Tensor:
        
        if (self.count % self.period) == (self.period - 1):
            permarray = torch.randperm(self.n)
            self.x = self.x[permarray]

        # get portions of data
        pos = np.random.randint(self.n-self.bs)
        x = self.x[pos:pos+self.bs].to(device)
        self.count += 1
        return x
