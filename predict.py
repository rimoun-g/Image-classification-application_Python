import argparse
import utility
import torch
import os
from os import path
import sys



# get user inputs for predict function
predict_parser = argparse.ArgumentParser (description = "predict image parameters")
predict_parser.add_argument('image_file',  help = 'enter the json file that contains the output classes')
predict_parser.add_argument('model_path',  help = 'enter the json file that contains the output classes')
predict_parser.add_argument('--labels_dic_file',  help = 'enter the json file that contains the output classes', default="cat_to_name.json")
predict_parser.add_argument('--topk',  help = 'enter the number top classes to be showed in the results', type=int, default=5)
predict_parser.add_argument("--device", help= "the training device one cpu or gpu", choices=["gpu","cpu"], default="cpu")


# setting variables for functions
args = predict_parser.parse_args()

image_path = args.image_file
model_path = args.model_path
labels_dic_file = args.labels_dic_file
topk = args.topk

# set the gpu/cpu variable
sel_dev = args.device

if sel_dev != "cpu" and torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def check_arguments():
    """This function checks the user inputs before using predict function"""
    if not path.isfile(image_path):
        print(image_path, "file cannot be found!")
        sys.exit()
    if not path.isfile(model_path):
        print(model_path, "model cannot be found!")
        sys.exit()
    if not path.isfile(labels_dic_file):
        print(labels_dic_file, "file cannot be found!")
        sys.exit()
    if  topk <=0:
        print("The topk cannot be less than 1")
        sys.exit()


# checking user inputs
check_arguments()

# loading labels json file as a dictionary
labels = utility.file_to_dic(labels_dic_file)

#_model, _model_info = utility.load_model()
#print(_model)
#print(_model_info)

# predicts the image and prints out the results
prediction = utility.predict(image_path, model_path, labels, topk, device)             
print(prediction)