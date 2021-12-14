import pytorch_lightning as pl

import torch

class Model(pl.LightningDataModule):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = torch.nn.Linear(64*64*3,100)

    def forward(self,x):
        x = x.view(1,-1)
        return self.layer1