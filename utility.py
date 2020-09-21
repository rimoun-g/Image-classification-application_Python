# importing needed libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
from PIL import Image
import json
import os

def data_processing(data_dir, train_batchsize = 64):
    """
    This function processes and transforms the data before training the model
    """
    print("Data processing has been initiated ...")
    
    # data folders
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # data and test sets transformation
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                     [0.229, 0.224, 0.225])])


    test_transforms = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=int(train_batchsize), shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=int(train_batchsize/2))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=int(train_batchsize/2))
    
  
    
    
    print("Data processing has been Done!")
    
    data_sets = {"train":train_dataset, "valid":valid_dataset, "test":test_dataset}
    data_loaders = {"train":train_dataloader, "valid":valid_dataloader, "test":test_dataloader}
   
    
    return data_sets, data_loaders
    
def file_to_dic(labels_dic_file):
    """This function loads a json file as a dictionary"""
    with open(labels_dic_file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
    
    
def model_build(arch = "vgg19", hidden_layer_size= 6000, output_size = 102):
    """This function builds the model structure"""
    if arch == "vgg19":
        model = models.vgg19(pretrained=True)
        input_size = 25088
    
    elif arch == "vgg16":
        model = models.vgg16(pretrained=True)
        input_size = 25088
   
    elif arch == "densenet121":
        model = models.densenet121(pretrained=True)
        input_size = 1024
    else:
        arch = "vgg19"
        model = models.vgg19(pretrained=True)
        input_size = 25088
    
    # Turning off the gradients of the model
    for prm in model.parameters():
        prm.requires_grad=False
    
    classifier = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(input_size, int(hidden_layer_size))),
                        ('relu1', nn.ReLU()),
                        ('dropout', nn.Dropout2d(p=0.5)),
                        ('output', nn.Linear(int(hidden_layer_size), int(output_size))),
                        ('logsoftmax', nn.LogSoftmax(dim=1))]))
    
    model.classifier = classifier
    
    return model


def crit_optim(model, lr = 0.001):
    """This function sets the learning rate value of the optimizer"""
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), float(lr))
    return criterion, optimizer

def loss_accuracy(valid_loader, model, criterion, device='cpu'):
    """This function calculates the accuracy of the train/validation/test set"""
    accurate = 0
    total = 0
    loss = 0
    model.eval()
    model.to(device)
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels =images.to(device), labels.to(device)
            outputs = model(images)
            imgs, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accurate += (predicted == labels).sum().item()
            loss += criterion(outputs, labels).item()
    return loss, accurate / total

def train_model(train_loader, valid_loader, model, criterion, optimizer, device='cpu',  epochs = 3, print_every=40):
    """This function trains the model as per the following parameters
    train_loader: the training data loaded and transformed by torch
    valid_loader: the validation data loaded and transformed by torch
    model: the model
    criterion: the criterion method
    optimizer: the chosen optimizer method
    device: trains the data using cpu or gpu it accepts cpu or cuda
    epochs: number of epochs
    print_every: show loss/validation statistics for each punch of steps
    """
    print("Model training has been initiated ...")
    steps = 0

   # uses the selected cpu/gpu for training
    
    model.to(device)

    for e in range(epochs):
        model.train()
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_loader):
            
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                # calculating validation loss & accuracy
                valid_loss, valid_accuracy = loss_accuracy(valid_loader, model, criterion, device)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss: {:.4f}".format(valid_loss),
                      "Validation Accuracy: {:.4f}".format(valid_accuracy))
                
                model.train()
                
                running_loss = 0
                
    print("Training has been done successfully!")
    
    return model


def save_model(model, train_dataset, optimizer, arch, output_classes, hidden_size, model_dir = "/"):
    """This function saves the model as a file to be loaded later for prediciton"""
    model_name = 'check_point_'+ arch +'.pth'
    
    # delete any existing file with the same name before saving 
    
    if os.path.exists(model_name):
        os.remove(model_name)
    
    if arch == "vgg19" or arch == "vgg16":
        input_size = 25088
    
    else:
        input_size = 1024

    classes = train_dataset.class_to_idx
    # saving the model 
    checkpoint = {'transfer_model': arch,
                  'class_to_idx': classes,
                  'features': model.features,
                  'classifier': model.classifier,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'input_size': input_size,
                  'output_size': output_classes,
                  'hidden_size': hidden_size
                 }
    
    torch.save(checkpoint, model_dir + "/" + model_name)
    print("the model has been saved successfully!")
    return  model_dir + "/" + model_name

def load_model(model_path):
    """This function loads a saved model"""
    # loadding the model
    model_info = torch.load(model_path)
    if model_info["transfer_model"] == "vgg19":
        model = models.vgg19(pretrained=True)
    elif model_info["transfer_model"] == "vgg16":
        model = models.vgg16(pretrained=True)
    elif model_info["transfer_model"] == "densenet121":
        model = models.densenet121(pretrained=True)
    else:
        model = models.vgg19(pretrained=True)
    # setting the model architecture 
    model.class_to_idx = model_info['class_to_idx'] 
    classifier = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(model_info['input_size'], model_info['hidden_size'])),
                        ('relu1', nn.ReLU()),
                        ('dropout', nn.Dropout2d(p=0.5)),
                        ('output', nn.Linear(model_info['hidden_size'], model_info['output_size'])),
                        ('logsoftmax', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    model.load_state_dict(model_info['state_dict'])
    print("Model has been loaded successfully!")
    return model, model_info

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    transform_img = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(244),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    img = transform_img(img)
    img = np.array(img)
    return img

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, labels_dict, topk, device = 'cpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    if device != 'cpu':
        if torch.cuda.is_available():
            device = 'cuda'
    
    with torch.no_grad():
        # open the image and process it
        img = process_image(image_path)
        # convert it to numpy tensors
        img = torch.from_numpy(img).to(device)
        img.unsqueeze_(0)
        img = img.float()
        # loading the model
        model, model_info = load_model(model)
        model.to(device)
        # predicting the output
        outputs = model(img)
        probs, indices = F.softmax(outputs).topk(topk)
        # returning the outputs as a dictionary of probalilites and classes indices
        
        probs = probs.to(device).numpy()
        indices  = indices.to(device).numpy()[0]
        idx_to_class = {v:k for k, v in model.class_to_idx.items()}
        classes = [idx_to_class[x] for x in indices]
        
        topk_lst = []
        for cls in classes:
            topk_lst.append(labels_dict[str(cls)])

        df_classes = pd.DataFrame({"probs":probs[0].tolist(), "class_idx": classes, "names":topk_lst})
        return df_classes
    
    
