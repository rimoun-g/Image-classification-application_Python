import argparse
import utility
import torch
import os
from os import path
import sys

# get user inputs for data processing and training functions
train_parser = argparse.ArgumentParser (description = "model training parameters")
train_parser.add_argument ('data_dir', help = 'enter the folder that contains the data')
train_parser.add_argument("--output_size", help= "the size of the output (number of classes)", default = 102, type=int)
train_parser.add_argument("--arch", help= "the model archtecture: vgg19 or vgg16 or densenet121", default = "vgg19", choices=["vgg19","vgg16","densenet121"])
train_parser.add_argument("--hidden_size", help= "the size of the hidden layer", default = 6000, type=int)
train_parser.add_argument("--lr", help= "the learning rate of the optimizing function", default =0.001 ,type=float)
train_parser.add_argument("--epochs", help= "the number of epochs", default =3, type=int)
train_parser.add_argument("--device", help= "the training device one cpu or gpu", choices=["gpu","cpu"], default="cpu")
train_parser.add_argument("--model_path", help= "the path for the created model to be saved in", default="/")

# setting variables for functions
args = train_parser.parse_args()
data_dir = args.data_dir
arch = args.arch
output = args.output_size
hidden_size = args.hidden_size
lr = args.lr
number_of_epochs = args.epochs

# set the gpu/cpu variable
sel_dev = args.device

if sel_dev != "cpu" and torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    
model_path = args.model_path

def check_arguments():
    """This function checks the user inputs before training"""
    if not path.isdir(data_dir):
        print(data_dir, "path cannot be found!")
        sys.exit()
    if not path.isdir(model_path):
        print(model_path, "path for saving model cannot be found!")
        sys.exit()
    if  hidden_size <=0 or hidden_size < output :
        print("The hidden size layer cannot be less than1 and cannot be less than the output classes!")
        sys.exit()
    if    lr <= 0 or lr >=1  :
        print("The learning rate value should be between 0 and 1")
        sys.exit()
    if number_of_epochs <= 0 or number_of_epochs > 10000:
        print("The number of epochs value should be between 1 and 10000")
        sys.exit()

check_arguments()     
print("The training will be processed using", device)

# data processing
datasets, loaders = utility.data_processing(data_dir)
# building model, criterion, and optimizer
model = utility.model_build(arch, hidden_size, output)
criterion, optimizer = utility.crit_optim(model, lr)
# training the model
model = utility.train_model(loaders["train"], loaders["valid"], model, criterion, optimizer, device, number_of_epochs, 40)
# print accuracy
__, test_accuracy  = utility.loss_accuracy(loaders["test"], model, criterion, device)
print("The model accuracy on the test set is: {:.2f}%".format(test_accuracy*100))
# save the model
model_name = utility.save_model(model, datasets["train"], optimizer, arch, output, hidden_size, model_dir = "/")
print("The model was saved in:", model_name)