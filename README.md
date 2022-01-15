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
``` 
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
``` 
### UWAGA!!!
Kod nie jest w całości nasz.
Kod opiera się o a także zawiera cudze fragmenty programów.
Źródła kodu użytego i wykorzystanego  w pracy znajdują się w komentarzach w odpowiednich plikach programu.
