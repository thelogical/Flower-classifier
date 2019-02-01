import matplotlib.pyplot as plt
import os
from torchvision import datasets,transforms,models
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from collections import OrderedDict
import argparse
import json
from torch.autograd import Variable
import warnings


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument("-a","--arch",metavar='',help="CNN architecture to use.Use \"-a help\" to see architecture list",default='vgg11')
parser.add_argument("-b","--batch_size",metavar='',help="Batch size for training",default=32,type=int)
parser.add_argument("-o","--optim",metavar='',help="Optimizer to use.Use \" -o help\" to see available optimizers",default='Adam')
parser.add_argument("-l","--loss",metavar='',help="loss function to use.Use \" -l help\" to list the loss functions",default='NLL')
parser.add_argument("-d","--dataset",metavar='',help="Path to custom dataset on which to train the network on",default="/home/workspace/ImageClassifier/flowers")
parser.add_argument("-r","--rate",metavar='',help="Learning rate(between 0 and 1) default is 0.0001",default=0.0001,type=float)
parser.add_argument("-e","--epoch",metavar='',help="No of epochs to train the network(minimum 1) default is 3",default=3,type=int)
parser.add_argument("-u","--hidden_units",metavar='',help="Number of hidden layer units in the classifier(minimum 1) default is 1000",default=1000,type=int)
parser.add_argument("-D","--device",metavar='',help="Either cpu or gpu")
args = parser.parse_args()
if(args.arch=='help'):
    print("List of available CNN networks:-")
    print("1. vgg11 (default)")
    print("2. vgg13")
    print("3. vgg16")
    print("4. vgg19")
    print("5. densenet121")
    print("6. alexnet")
    quit()
if(args.optim=='help'):
    print("List of available optimizers:-")
    print("1. SGD -> Stochastic Gradient Descent")
    print("2. Adaddelta -> Adaptive learning Algorithm")
    print("3. Adagrad -> Adagrad Algorithm")
    print("4. Adam (default)")
    print("5. RMS -> RMSprop algorithm")
    print("6. RPROP -> Resilient Backpropogation algorithm")
    quit()
if(args.loss=='help'):
    print("List of available loss functions:-")
    print("1. L1 -> Mean absolute error")
    print("2. NLL -> Negative Log Likelihood loss(default)")
    print("3. Poisson -> NLL loss with poisson distribution")
    print("4. MSE -> Mean Squared Error loss")
    print("5. Cross -> CrossEntropyLoss")
    quit()

if(not(args.rate>0 and args.rate<1)):
    print("Error: Invalid learning rate")
    print("Must be between 0 and 1 exclusive")
    quit()

if(args.batch_size < 0):
    print("Error: Invalid batch size")
    print("Must be greater than 0")
    quit()
    
if(args.epoch <= 0):
    print("Error: Invalid epoch value")
    print("Must be greater than 0")
    quit()
    
if(args.hidden_units <=0):
    print("Error: Invalid number of hidden units given")
    print("Must be greater than 0")
    quit()
    
archs = ["vgg11","vgg13","vgg16","vgg19","alexnet","densenet"]
optims = ["SGD","Adadelta","Adagrad","Adam","RMS","RMSprop"]
loss_f = ["L1","NLL","Poisson","MSE","Cross"]

if args.arch not in archs:
    print("Error: Invalid architecture name received")
    print("Type \"python train.py -a help\" for more information")
    quit()
    
if args.optim not in optims:
    print("Error: invalid optimizer name received")
    print("Type \"python train.py -o help\" for more information")
    quit()
    
if args.loss not in loss_f:
    print("Error: invalid loss function name received")
    print("type \"python train.py -l help\" for more information")
    quit()

if args.device not in ['cpu','gpu']:
    print("Error: invalid device name received")
    print("It must be either 'cpu' or 'gpu'")
    quit()
    
if args.device=='gpu':
    args.device='cuda'
    
    
#printing all the arguments to STDOUT for verification

print("Architecture-> "+args.arch)
print("Optimizer-> "+args.optim)
print("Loss functoin-> "+args.loss)
print("Batchsize-> "+str(args.batch_size))
print("Learning rate-> "+str(args.rate))
print("epoch limit ->"+str(args.epoch))
print("hidden units ->"+str(args.hidden_units))
if(args.device=='cuda'):
    print("Device-> "+torch.cuda.get_device_name(0))
else:
    print("Device-> "+args.device)



data_dir = args.dataset
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]) 

# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir,transform=train_transforms)
validation_data = datasets.ImageFolder(valid_dir,transform=validation_transforms)
test_data = datasets.ImageFolder(test_dir,transform=test_transforms) 

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
validationloader = torch.utils.data.DataLoader(validation_data, batch_size=16)
testloader = torch.utils.data.DataLoader(test_data, batch_size=16)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# TODO: Build and train your network

if args.arch=='vgg11':
    model = models.vgg11(pretrained=True)
    fc1_in_units=25088
elif args.arch=='vgg13':
    model = models.vgg13(pretrained=True)
    fc1_in_units=25088
elif args.arch=='vgg16':
    model = models.vgg16(pretrained=True)
    fc1_in_units=25088
elif args.arch=='vgg19':
    model = models.vgg19(pretrained=True)
    fc1_in_units=25088
elif args.arch=='alexnet':
    model = models.alexnet(pretrained=True)
    fc1_in_units=9216
else:
    model = models.densenet121(pretrained=True)
    fc1_in_units=1024                                 #Not sure about this value
    
model.name=args.arch
    
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(fc1_in_units, args.hidden_units)),
                          ('relu', nn.ReLU()),
                          ('drop',nn.Dropout(0.2)),
                          ('fc2', nn.Linear(args.hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

model.classifier = classifier

if(args.optim=='SGD'):
    optimizer = optim.SGD(model.classifier.parameters(), lr=args.rate)
elif(args.optim=='Adadelta'):
    optimizer = optim.Adadelta(model.classifier.parameters(), lr=args.rate)
elif(args.optim=='Adagrad'):
    optimizer = optimizer = optim.Adagrad(model.classifier.parameters(), lr=args.rate)
elif(args.optim=='Adam'):
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.rate)
elif(args.optim=='RMS'):
    optimizer = optim.RMS(model.classifier.parameters(), lr=args.rate)
else:
    optimizer = optim.Rprop(model.classifier.parameters(), lr=args.rate)
    
if args.loss=='L1':
    criterion = nn.L1Loss()
elif args.loss=='NLL':
    criterion = nn.NLLLoss()
elif args.loss=='Poisson':
    criterion = nn.PoissonNLLoss()
elif args.loss=='MSE':
    criterion = nn.MSELoss()
else:
    criterion = nn.CrossEntropyLoss()
    
optimizer.zero_grad()
model.classifier = classifier
epochs = args.epoch
print_every = 4
steps = 0

model.to(args.device)

for e in range(epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate(trainloader):
        steps += 1
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if steps % print_every == 0:
            print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Loss: {:.4f}".format(running_loss/print_every))
            
            running_loss = 0
    
    correct = 0
    total = 0
    print("Validating the net after epoch {}/{}".format(e+1,epochs))
    for ii, (inputs, labels) in enumerate(validationloader):
        with torch.no_grad():
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            # Forward and backward passes
            outputs = model.forward(inputs)
            _, predicted = torch.max(torch.exp(outputs.data), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy-> {0:.2f}".format(correct/total*100))

print("Validating the net using Test Images")
for ii, (inputs, labels) in enumerate(testloader):
    with torch.no_grad():
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = model.forward(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print("Accuracy-> {0:.2f}".format(correct/total*100))
 

print("Training complete")
print("Saving checkpoint.... ")
model.loss = loss
model.epoch = epochs
torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
             'model': args.arch,
             'hidden_units': args.hidden_units,
             'class_to_idx': train_data.class_to_idx,
             'fc1_in_units': fc1_in_units
            }, '/home/workspace/modl.pth')
print("Checkpoint saved as /home/workspace/modl.pth.")
print("Done")
