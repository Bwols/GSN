import os
import torch
from tests import show_results_of_model, load_data, load_model, calc_accuracy
from PIL import Image
import torchvision
import torchvision.transforms as transforms
imsize = 256
loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])

transform = transforms.Compose(
            [
             transforms.ToTensor(),
             transforms.Normalize((0,), (1,)),  # zakres 0,1
             ])

def label_student():
    directory = os.fsencode("models")
    path_to_model = str(directory)[2:-1] + "\\" +  str(os.listdir(directory))[3:-2] 

    model = load_model(path_to_model)
    model = model.cuda()
    model.eval()

    image = image_loader("Pokemon_Images_Student/1.png")
    model(image)
def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = transform(image).to('cuda').unsqueeze(0)
    return image

label_student()
