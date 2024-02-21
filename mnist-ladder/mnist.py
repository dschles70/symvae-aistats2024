import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms

# this is just a wrapper for loading, organizing mini-batches etc.
class MNIST():
    def __init__(self,
                 bs : int):
        
        self.bs = bs

        # load data
        train_set = torchvision.datasets.MNIST("./data", download=True, train=True, transform=transforms.ToTensor())
        test_set = torchvision.datasets.MNIST("./data", download=True, train=False, transform=transforms.ToTensor()) 

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100)

        x = []
        for images, _ in train_loader:
            x += [images]
        for images, _ in test_loader:
            x += [images]

        x = torch.stack(x).flatten(0, 1)
        self.x = (x>0.4375).float()

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
