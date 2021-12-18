from tests import show_results_of_model, load_data, load_model

path_to_model = "C:\\Users\\JeLoń\\Desktop\\GSN\\lightning_logs\\version_121\\checkpoints\\epoch=4-step=2139.ckpt"
#path_to_model = "models/first_test.ckpt"
model = load_model(path_to_model)

data = load_data(16)

show_results_of_model(model,data)