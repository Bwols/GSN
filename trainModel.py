import pytorch_lightning as pl
from model import Model
from resnet import ResNet
import torch
from torch.nn import functional as F
from dataset import DataLoader

def OneHot(label):
    return torch.zeros(150)
def ResNet50():
    return ResNet([3, 4, 6, 3])
def ResNet152():
    return ResNet([3, 8, 36, 3])
    
class PokemonClassifier(pl.LightningModule):

    def __init__(self):
        super(PokemonClassifier, self).__init__()
        self.layer1 = torch.nn.Linear(64 * 64 * 3, 150)
        #self.model = Model()
        self.model = ResNet50()

    def cross_entropy_loss(self, logits, labels):
        loss = torch.nn.BCELoss()
        #return F.nll_loss(logits, labels)
        return loss(logits.float(),labels.float())


    def forward(self,x):
        batch_size = x.shape[0]
        x = x.view(batch_size,-1)
        x = self.layer1(x)
        x = torch.sigmoid(x)
        return x


    def training_step(self, train_batch, batch_idx):

        x, label, name = train_batch
        label = F.one_hot(label,150)
        print(x.size())
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, label)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



model = PokemonClassifier()


train_dataloader = DataLoader().get_data_loader()
#val_loader = DataLoader().get_data_loader()
trainer = pl.Trainer(max_epochs=10,gpus=1)

trainer.fit(model, train_dataloader)