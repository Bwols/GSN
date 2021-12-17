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
        loss = torch.nn.BCELoss(reduction='sum')
        #loss = torch.nn.L1Loss(reduction='sum')
        # return F.nll_loss(logits, labels)
        return loss(logits.float(), labels.float())



    def forward(self, x):
        x = self.model(x)

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
        x, labels, name = val_batch
        labels_cp = F.one_hot(labels, 150)
        logits = self.forward(x)

        class_propabilities, pred_labels = torch.max(logits, 1)
        right_preds = 0

        for i in range(0, len(pred_labels)):
            if pred_labels[i] == labels[i]:
                right_preds += 1
        print("     Right predictions{}/{}".format(right_preds,len(pred_labels)))

        loss = self.cross_entropy_loss(logits, labels_cp)



        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)#lr=1e-3
        return optimizer

