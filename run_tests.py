from tests import show_results_of_model, load_data, load_model, calc_accuracy

#path_to_model = "C:\\Users\\JeLoń\\Desktop\\GSN\\lightning_logs\\version_121\\checkpoints\\epoch=4-step=2139.ckpt"

def graphic_results():
    path_to_model = "models/first_test_new.ckpt"
    model = load_model(path_to_model)

    data = load_data(8)

    show_results_of_model(model, data, "run1.png")

#graphic_results()

calc_accuracy("models/first_test_new.ckpt")
