# GSN


## Przykładowe użycie:

``` 
python3.7 main.py +name=run1 architecture_config=resnet50_256

```

## Podstawowe katalogi config

* `dataset_config` -- 
* `training_config` -- 
* `architecture_config` -- 

### Przykładowe ustawienia uczenia

dataset_config:
  dir: pokemon_images
  pokedex: pokedex.csv
training_config:
  batch_size: 16
  lr: 0.001
  epochs: 25
architecture_config:
  architecture: ResNet50
  image_size: 64

