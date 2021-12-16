import pytorch_lightning as pl
from model import PokemonClassifier

from dataset import DataLoader



model = PokemonClassifier()


train_dataloader = DataLoader().get_data_loader()
val_loader = DataLoader(batch_size=16).get_data_loader()

trainer = pl.Trainer(max_epochs=1,gpus=1)

trainer.fit(model, train_dataloader, val_loader)

