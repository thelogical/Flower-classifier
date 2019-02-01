import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets,transforms,models
import torch
import torch.nn as nn
from matplotlib.pyplot import imshow 
from PIL import Image
import numpy as np
import argparse
from collections import OrderedDict
import json
import warnings


warnings.filterwarnings("ignore")

means = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
def process_image(image):
  im = Image.open(image)
  im.thumbnail((256,256))
  width, height = im.size   
  left = (width - 224)/2
  top = (height - 224)/2
  right = (width + 224)/2
  bottom = (height + 224)/2
  im = im.crop((left, top, right, bottom))
  img = np.asarray(im)
  img = img/255
  img = (img - means)/std
  return img.transpose(2,0,1)

parser = argparse.ArgumentParser()

parser.add_argument("-c","--checkpoint",metavar='',help="Path of the checkpoint to load",default='/home/workspace/modl.pth')
parser.add_argument("-p","--path",metavar="",help="Absolute path of the image to be classified",default='/home/workspace/ImageClassifier/flowers/test/1/image_06743.jpg')
parser.add_argument("-D","--device",metavar='',help="Either cpu or gpu")
parser.add_argument("-k","--topk",metavar='',help="The number of species to display having highest probabilities",default=5,type=int)
parser.add_argument("-j","--JSON",metavar='',help="Path to the JSON file to be used for class mapping",default="/home/workspace/ImageClassifier/cat_to_name.json")
args = parser.parse_args()

with open(args.JSON, 'r') as f:
    cat_to_name = json.load(f)

if(args.device not in ['cpu','gpu']):
    print("Error: Incorrect device name received")
    quit()

if(args.device=='gpu'):
    args.device='cuda'

if(args.device=='cpu'):
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
else:
    checkpoint = torch.load(args.checkpoint)

def predict(path,checkpoint,k_values=5):
    
    name = checkpoint['model']
    hidden_units = checkpoint['hidden_units']
    fc1_in_units = checkpoint['fc1_in_units']
    print("Model Detected-> "+name)
    print("Hidden_layers-> "+str(hidden_units))

    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(fc1_in_units, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('drop',nn.Dropout(0.2)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    if name=='vgg11':
        model = models.vgg11(pretrained=True)
    elif name=='vgg13':
       model = models.vgg13(pretrained=True)
    elif name=='vgg16':
        model = models.vgg16(pretrained=True)
    elif name=='vgg19':
        model = models.vgg19(pretrained=True)
    elif name=='alexnet':
        model = models.alexnet(pretrained=True)
    else:
        model = models.densenet121(pretrained=True)
    
    model.classifier = classifier

    for param in model.parameters():
        param.requires_grad = False  
     
    print("Building model...")
    model.load_state_dict(checkpoint['model_state_dict'])  
    print("Done")
    
    model.to(args.device)
    model.classifier = classifier
    model.eval()
    
    print("Processing Image")
    img = process_image(args.path)
    print("Done")
    img = torch.Tensor(img)
    img = img.to(args.device)
    model.to(args.device)
    img.unsqueeze_(0)
    print("Predicting image now..")
    out = model(img)
    out = torch.exp(out)
    k = out.topk(k_values)
    if(args.device=='cuda'):
        probabilities = k[0][0].cpu().detach().numpy()
    else:
        probabilities = k[0][0].detach().numpy()
    
    idx_to_class = {v: k for k, v in checkpoint['class_to_idx'].items()}
    objs = [idx_to_class[int(i)] for i in k[1][0]]
    objects = [cat_to_name[str(i)] for i in objs]
    x_pos = np.arange(k_values)
    plt.barh(x_pos, probabilities, align='center', alpha=1)
    plt.yticks(x_pos, objects)
    plt.xlabel('Probability')
    plt.title('Flower species prediction')
    plt.savefig('/home/workspace/prediction.png')
    print("Predictions saved as /home/workspace/prediction.png")
    
predict(args.path,checkpoint,args.topk)
