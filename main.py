import hydra
from omegaconf import DictConfig, OmegaConf

CONFIG_PATH = "config"
CONFIG_NAME = "default_config"


@hydra.main(config_path=CONFIG_PATH,config_name=CONFIG_NAME)
def use_hydra(config):
    print(OmegaConf.to_yaml(config))
    dataset_dir = config.dataset_config.dir
    dataset_pokedex = config.dataset_config.pokedex

    print(dataset_pokedex)

use_hydra()