import torch 
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch import nn
import torch.optim as optim

import numpy as np

def load_data(data_path):
    
    data_dir = data_path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.RandomResizedCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    return trainloader, validloader, testloader, train_data

def network(structure='vgg16', dropput=0.05, hidden_layer1= 4096, learning_rate=0.0003, gpu='cuda:0'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'vgg13':
        model = models.vgg13(pretrained=True)
        
    
    for params in model.parameters():
        params.requires_grad = False
        
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088, hidden_layer1)),
                            ('relu1', nn.ReLU()),
                            ('dropout1', nn.Dropout(p=dropput)),
                            ('fc2', nn.Linear(hidden_layer1, 512)),
                            ('relu2', nn.ReLU()),
                            ('dropout2', nn.Dropout(p=dropput)),
                            ('fc3', nn.Linear(512, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                        ]))
    
    model.classifier = classifier

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device)
    
    return model, criterion, optimizer


def train_network(model, criterion, optimizer, trainloader, gpu, epochs, validloader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print_every = 5
    steps = 0
    running_loss = 0
    
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                      f"Valid accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
            
    return model, optimizer

def test_network(model, testloader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    acc = 0
    total = 0
    model.to(device)

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, pred = torch.max(outputs.data, 1)

            total += labels.size(0)
            acc += (pred == labels).sum().item()

    print("{}%".format(round(100 * acc / total,3)))
    

def save_model(save_dir, model, epochs, optimizer, train_data, arch, hidden_layer, dropout):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'input_size':25088,
                 'output_size':102,
                 'arch': arch,
                 'dropout': dropout,
                 'hidden_layer': hidden_layer,
                 'state_dict': model.state_dict(),
                 'class_to_idex': model.class_to_idx,
                 'opt_state': optimizer.state_dict,
                 'num_epochs': epochs}

    torch.save(checkpoint, save_dir)
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    arch = checkpoint['arch']
    dropout = checkpoint['dropout']
    hidden_layer = checkpoint['hidden_layer']
    
    model, criterion, optimizer = network(arch, dropout, hidden_layer)
    model.class_to_idx = checkpoint['class_to_idex']
    model.load_state_dict(checkpoint['state_dict'])
    
    
    return model

def predict(image_path, model, topk=5, gpu='cuda:0'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    load_model = load_checkpoint(model)
    img = process_image(image_path)
    img_add_dim = img.unsqueeze_(0)

    load_model.eval()
    with torch.no_grad():
        output = load_model.forward(img_add_dim)

    probs = torch.exp(output)
    probs_top = probs.topk(topk)[0]
    index_top = probs.topk(topk)[1]
    
    probs_top_list = np.array(probs_top)[0]
    index_top_list = np.array(index_top[0])
    
    class_to_idx = load_model.class_to_idx
    indx_to_class = {x: y for y, x in class_to_idx.items()}

    classes_top_list = []
    for index in index_top_list:
        classes_top_list += [indx_to_class[index]]
        
    return probs_top_list, classes_top_list