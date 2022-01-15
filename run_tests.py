from tests import show_results_of_model, load_data, load_model, calc_accuracy
from dataset import DataLoader




model = load_model(path_to_model="models/exp1_ResNet101_64.ckpt",
                   architecture="ResNet101",
                   image_size=64,device="cuda")


test_dataloader = DataLoader(dataset_dir="Pokemon_Images",
                             labels_csv="pokedex.csv" ,
                             batch_size=16, shuffle=True, max_size=100000,
                             augmentation=False).get_data_loader()

calc_accuracy(model=model, test_dataloader=test_dataloader,device="cuda")

#TODO odkomentować dla graficznego rezultatu
#data = load_data(dataset_dir="Pokemon_Images",labels_csv="pokedex.csv" , batch_size=8,max_size=9,augmentation=False)

#show_results_of_model(model, data, "example.png")

