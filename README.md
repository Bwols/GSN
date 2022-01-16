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
Do stworzenia kodu zostały użyte następujące źródła:
https://developpaper.com/pytorch-implementation-examples-of-resnet50-resnet101-and-resnet152/ - Model resnetu

https://medium.com/decathlontechnology/improving-performance-of-image-classification-models-using-pretraining-and-a-combination-of-e271c96808d2

https://discuss.pytorch.org/t/how-to-classify-single-image-using-loaded-net/1411

https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09

https://towardsdatascience.com/keeping-up-with-pytorch-lightning-and-hydra-2nd-edition-34f88e9d5c90

https://www.delftstack.com/howto/python/python-read-csv-into-array/

https://hydra.cc/docs/tutorials/intro/

https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5


Źródło datasetu: https://www.kaggle.com/lantian773030/pokemonclassification

