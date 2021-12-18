import pytorch_lightning as pl
from model import PokemonClassifier

from dataset import DataLoader
from data_preprocess import make_dir
MODELS_DIR = "models"


def train_model(train_name,batch_size=16, max_epochs=5):
    make_dir(MODELS_DIR)

    model = PokemonClassifier()


    train_dataloader = DataLoader(batch_size=batch_size,shuffle=True).get_data_loader()
    val_loader = DataLoader(batch_size=64,shuffle=True).get_data_loader()

    trainer = pl.Trainer(max_epochs=max_epochs)

    trainer.fit(model, train_dataloader)#

    make_dir(MODELS_DIR)
    trainer.save_checkpoint("{}/{}.ckpt".format(MODELS_DIR, train_name))




train_model("first_test",batch_size=16)