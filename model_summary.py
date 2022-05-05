import torch
from torch.autograd import Variable
from groupy.gconv.pytorch_gconv import P4ConvZ2, P4ConvP4, P4MConvZ2, P4MConvP4M
import torchvision
from torchvision.transforms import ToTensor
from torchvision import datasets, models, transforms
import numpy as np
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import nn
from torchsummary import summary
import time
import copy
import matplotlib.pyplot as plt
import os

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        channel_size = [64,32,16,16,32,64]
        strides = [1,1,2,1,1,1,1]
        kernels = [3,3,3,5,5,7,7]
        self.p4convz2 = nn.ModuleList()
        self.p4mconvz2 = nn.ModuleList()
        self.n_layers = len(channel_size)    
        
        for layer in range(len(channel_size)):
            oc = channel_size[layer]
            s = strides[layer]
            k = kernels[layer]
            p = int(k//2)
            if layer == 0:
                self.p4convz2.extend([P4ConvZ2(in_channels=1, out_channels=oc, kernel_size=k, stride=s, padding=p)])
                self.p4mconvz2.extend([P4MConvZ2(in_channels=1, out_channels=oc, kernel_size=k, stride=s, padding=p)])
            else:
                ic = channel_size[layer-1]*12
                self.p4convz2.extend([P4ConvZ2(in_channels=ic, out_channels=oc, kernel_size=k, stride=s, padding=p)])
                self.p4mconvz2.extend([P4MConvZ2(in_channels=ic, out_channels=oc, kernel_size=k, stride=s, padding=p)])
                
                
        self.relu = nn.ReLU()
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(14*14*channel_size[-1]*12, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        
        for layer in range(self.n_layers):
            x2 = self.p4convz2[layer](x)
            x3= self.p4mconvz2[layer](x)
            x2 = x2.view(x2.shape[0],x2.shape[1]*x2.shape[2], x2.shape[3],x2.shape[4])
            x3 = x3.view(x3.shape[0],x3.shape[1]*x3.shape[2], x3.shape[3],x3.shape[4])
            x = torch.cat((x2,x3),1)
            x = self.relu(x)

        x = torch.flatten(x,1)
        logits = self.linear_relu_stack(x)
        return logits



if __name__ == "__main__":
    fn_model = 'temp/model_g_4_test.pt'
    data_transformation = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation([-10,10]),
        transforms.ToTensor(),
        transforms.Normalize(0,1)
    ]),
    'val':
    ToTensor()
    }
    train_ds = torchvision.datasets.FashionMNIST('dataset',download=True,train=True,
                                                transform=data_transformation['train'])
    test_ds = torchvision.datasets.FashionMNIST('dataset',download=True,train=False,
                                                transform=data_transformation['val'])

    # os.environ['CUDA_VISIBLE_DEVICES']='0,1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_datasets = {'train':train_ds, 'val':test_ds}
    dataset_sizes = {x: len(img_datasets[x]) for x in ['train', 'val']}

    model = NeuralNetwork()
    model = model.to(device)

    try:
        model.load_state_dict(torch.load(fn_model))
        optimizer_ft = optim.Adam(model.parameters(), lr=0.001)#, momentum=0.9)
        best_model_wts = copy.deepcopy(model.state_dict())
    except:
        optimizer_ft = optim.Adam(model.parameters(), lr=0.001)#, momentum=0.9)
        best_model_wts = copy.deepcopy(model.state_dict())

    print(summary(model, (1,28,28)))
    # dataloaders = {x: torch.utils.data.DataLoader(img_datasets[x], batch_size=200, shuffle=True, num_workers=1) for x in ['train','val']}

    # criterion = nn.CrossEntropyLoss()

    # # Observe that all parameters are being optimized
    

    # # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # scheduler =exp_lr_scheduler
    # optimizer=optimizer_ft

    # num_epochs=30

    # since = time.time()

    
    # best_acc = 0.0
    # losses = []


    # for epoch in range(num_epochs):
    #     print(f'Epoch {epoch}/{num_epochs - 1}')
    #     print('-' * 10)

    #     # Each epoch has a training and validation phase
    #     for phase in ['train', 'val']:
    #         if phase == 'train':
    #             model.train()  # Set model to training mode
    #         else:
    #             model.eval()   # Set model to evaluate mode

    #         running_loss = 0.0
    #         running_corrects = 0

    #         # Iterate over data.
    #         for inputs, labels in dataloaders[phase]:
    #             inputs = inputs.to(device)
    #             labels = labels.to(device)

    #             # zero the parameter gradients
    #             optimizer.zero_grad()

    #             # forward
    #             # track history if only in train
    #             with torch.set_grad_enabled(phase == 'train'):
    #                 outputs = model(inputs)
    #                 _, preds = torch.max(outputs, 1)
    #                 loss = criterion(outputs, labels)
    #                 losses.append(loss)
    #                 # backward + optimize only if in training phase
    #                 if phase == 'train':
    #                     loss.backward()
    #                     optimizer.step()

    #             # statistics
    #             running_loss += loss.item() * inputs.size(0)
    #             running_corrects += torch.sum(preds == labels.data)
    #         if phase == 'train':
    #             # pass
    #             scheduler.step()

    #         torch.save(model.state_dict(), fn_model)
    #         epoch_loss = running_loss / dataset_sizes[phase]
    #         epoch_acc = running_corrects.double() / dataset_sizes[phase]

    #         print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    #         # deep copy the model
    #         if phase == 'val' and epoch_acc > best_acc:
    #             best_acc = epoch_acc
    #             best_model_wts = copy.deepcopy(model.state_dict())

    #     print()

    #     time_elapsed = time.time() - since
    #     print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    #     print(f'Best val Acc: {best_acc:4f}')

    #     # load best model weights
    #     model.load_state_dict(best_model_wts)

    #     torch.save(model.state_dict(), fn_model)