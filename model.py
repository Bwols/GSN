import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from resnet import ResNet

def OneHot(label):
    return torch.zeros(150)


def ResNet50():
    return ResNet([3, 4, 6, 3], inputSize=64)


def ResNet152():
    return ResNet([3, 8, 36, 3], inputSize=64)


class PokemonClassifier(pl.LightningModule):

    def __init__(self):
        super(PokemonClassifier, self).__init__()

        # self.model = Model()
        self.model = ResNet50()

    def cross_entropy_loss(self, logits, labels):
        loss = torch.nn.BCELoss()
        # return F.nll_loss(logits, labels)
        return loss(logits.float(), labels.float())

    def forward(self, x):
        x = self.model(x)
        x = torch.sigmoid(x)
        return x

    def training_step(self, train_batch, batch_idx):
        x, label, name = train_batch
        label = F.one_hot(label, 150)
        # print(x.size())
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, label, name = val_batch
        label = F.one_hot(label, 150)
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, label)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

