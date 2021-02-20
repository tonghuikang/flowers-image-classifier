import numpy as np
import time
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models

class homemade_CNN(nn.Module):
    def __init__(self, num_layers=1):
        super(homemade_CNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.num_layers = num_layers
        for i in range(self.num_layers):
            setattr(self, "layer{}".format(i+2), nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
                nn.Dropout2d(0.01),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            ))
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=193600, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        
    def forward(self, x):
        out = self.layer1(x)
        for i in range(self.num_layers):
            out = getattr(self, "layer{}".format(i+2))(out)
        out = self.layer0(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        
        return out

homemade_CNN_small = homemade_CNN(0)
homemade_CNN_large = homemade_CNN(10)
setattr(models, "homemade_CNN_small", homemade_CNN_small)
setattr(models, "homemade_CNN_large", homemade_CNN_large)


# Define classifier class
class NN_Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
        '''
        super().__init__()
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)

# Define validation function 
def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
                
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy

# Define NN function
def make_NN(n_hidden, n_epoch, labelsdict, lr, device, model_name, trainloader, validloader, train_data,
            not_use_pretrained=False, train_all_layers=False):
    
    if not_use_pretrained:
        model = getattr(models, model_name)
    else:
    # Import pre-trained NN model 
        model = getattr(models, model_name)(pretrained=not not_use_pretrained)
    
    if not train_all_layers:
    # Freeze parameters that we don't need to re-train 
        for param in model.parameters():
            param.requires_grad = False

    # Make classifier
    name_of_last_layer = list(model.named_modules())[-1][0]
    n_in = list(model.children())[-1].in_features
    n_out = len(labelsdict)
    setattr(model, name_of_last_layer, NN_Classifier(input_size=n_in, output_size=n_out, hidden_layers=n_hidden))
    
    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(getattr(model, name_of_last_layer).parameters(), lr = lr)

    model.to(device)
    start = time.time()

    steps = 0 
    running_loss = 0
    
    prev_step = 0
    prev_epoch = -1
    special_steps = set([1,2,5,10,20,51])
    
    for e in range(n_epoch+1):
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            steps += 1

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if prev_epoch != e or steps in special_steps:
                # Eval mode for predictions
                model.eval()

                # Turn off gradients for validation
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion, device)

                print("Epoch: {}/{} Step: {} - ".format(e, n_epoch, steps),
                      "Training Loss: {:.3f} - ".format(running_loss/(steps-prev_step)),
                      "Validation Loss: {:.3f} - ".format(test_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
                running_loss = 0
                prev_step = steps
                prev_epoch = e
                if e == n_epoch:
                    break

                # Make sure training is back on
                model.train()
    
    print('model:', model_name, '- hidden layers:', n_hidden, '- epochs:', n_epoch, '- lr:', lr)
    print(f"Run time: {(time.time() - start)/60:.3f} min")
    return model

# Define function to save checkpoint
# not for non-densenet
def save_checkpoint(model, path):
    checkpoint = {'c_input': model.classifier.n_in,
                  'c_hidden': model.classifier.n_hidden,
                  'c_out': model.classifier.n_out,
                  'labelsdict': model.classifier.labelsdict,
                  'c_lr': model.classifier.lr,
                  'state_dict': model.state_dict(),
                  'c_state_dict': model.classifier.state_dict(),
                  'opti_state_dict': model.classifier.optimizer_state_dict,
                  'model_name': model.classifier.model_name,
                  'class_to_idx': model.classifier.class_to_idx
                  }
    torch.save(checkpoint, path)
    
# Define function to load model
# not for non-densenet
def load_model(path):
    cp = torch.load(path)
    
    # Import pre-trained NN model 
    model = getattr(models, cp['model_name'])(pretrained=True)
    
    # Freeze parameters that we don't need to re-train 
    for param in model.parameters():
        param.requires_grad = False
    
    # Make classifier
    model.classifier = NN_Classifier(input_size=cp['c_input'], output_size=cp['c_out'], \
                                     hidden_layers=cp['c_hidden'])
    
    # Add model info 
    model.classifier.n_in = cp['c_input']
    model.classifier.n_hidden = cp['c_hidden']
    model.classifier.n_out = cp['c_out']
    model.classifier.labelsdict = cp['labelsdict']
    model.classifier.lr = cp['c_lr']
    model.classifier.optimizer_state_dict = cp['opti_state_dict']
    model.classifier.model_name = cp['model_name']
    model.classifier.class_to_idx = cp['class_to_idx']
    model.load_state_dict(cp['state_dict'])
    
    return model

def test_model(model, testloader, device='cuda'):  
    model.to(device)
    model.eval()
    accuracy = 0
    
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
                
        output = model.forward(images)
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    print('Testing Accuracy: {:.3f}'.format(accuracy/len(testloader)))
    

