import pytorch_lightning as pl
from model import PokemonClassifier

from dataset import DataLoader

from data_preprocess import make_dir

from data_preprocess import make_dir
MODELS_DIR = "teacher_models"
TEACHER_DIR = "Pokemon_Images_Teacher"
TEACHER_CSV = "pokedex_teacher.csv"
from tests import show_results_of_model, load_data

def train_model(train_name,batch_size=16, max_epochs=5):
    make_dir(MODELS_DIR)

    model = PokemonClassifier()

    train_dataloader = DataLoader(batch_size=batch_size,shuffle=True).get_data_loader()
    val_loader = DataLoader(batch_size=64,shuffle=False).get_data_loader()


    trainer = pl.Trainer(max_epochs=max_epochs,gpus=1)

    trainer.fit(model, train_dataloader, val_loader)#

    make_dir(MODELS_DIR)
    trainer.save_checkpoint("{}/{}.ckpt".format(MODELS_DIR, train_name))

    data = load_data(16)
    show_results_of_model(model, data, output_file="in_train.png")


train_model("first_test",batch_size=16, max_epochs=5)
