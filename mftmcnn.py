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
import pickle

from mftma.manifold_analysis_correlation import manifold_analysis_corr
from mftma.utils.make_manifold_data import make_manifold_data
from mftma.utils.activation_extractor import extractor
from mftma.utils.analyze_pytorch import analyze


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        channel_size = [64,32,16,16,32,64]
        strides = [1,1,2,1,1,1,1]
        kernels = [3,3,3,5,5,7,7]
        self.conv2d = nn.ModuleList()
        self.n_layers = len(channel_size)    
        
        for layer in range(len(channel_size)):
            oc = channel_size[layer]
            s = strides[layer]
            k = kernels[layer]
            p = int(k//2)
            if layer == 0:
                self.conv2d.extend([nn.Conv2d(1, oc*13, kernel_size=(k, k), stride=(s, s), padding=(p, p), bias=False)]) 

            else:
                ic = channel_size[layer-1]*13
                self.conv2d.extend([nn.Conv2d(ic, oc*13, kernel_size=(k, k), stride=(s, s), padding=(p, p), bias=False)])

                
                
        self.relu = nn.ReLU()
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(14*14*channel_size[-1]*13, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        
        for layer in range(self.n_layers):
            x = self.conv2d[layer](x)
            x = self.relu(x)

        x = torch.flatten(x,1)
        logits = self.linear_relu_stack(x)
        return logits


if __name__ == "__main__":
    fn_model = 'temp/model_cnn_4.pt'
    fn_fig = 'temp/model_cnn_4_mftm.png'
    fn_mftm_results = 'temp/mftm_cnn_results.pkl'
    num_epochs=30


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

    os.environ['CUDA_VISIBLE_DEVICES']='0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_datasets = {'train':train_ds, 'val':test_ds}
    dataset_sizes = {x: len(img_datasets[x]) for x in ['train', 'val']}

    
    try:
        model = NeuralNetwork()
        model = nn.DataParallel(model, device_ids=[0])
        model = model.to(device)
        model.load_state_dict(torch.load(fn_model))
        optimizer_ft = optim.Adam(model.parameters(), lr=0.001)#, momentum=0.9)
        best_model_wts = copy.deepcopy(model.state_dict())
    except:
        model = NeuralNetwork()
        model = nn.DataParallel(model, device_ids=[0])
        model = model.to(device)
        optimizer_ft = optim.Adam(model.parameters(), lr=0.001)#, momentum=0.9)
        best_model_wts = copy.deepcopy(model.state_dict())

    dataloaders = {x: torch.utils.data.DataLoader(img_datasets[x], batch_size=200, shuffle=True, num_workers=1) for x in ['train','val']}

    criterion = nn.CrossEntropyLoss()

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    scheduler =exp_lr_scheduler
    optimizer=optimizer_ft

    since = time.time()

    best_acc = 0.0
    losses = []


    model = model.eval()

    sampled_classes = 10
    examples_per_class = 200

    datas = make_manifold_data(train_ds, sampled_classes, examples_per_class, seed=0)
    datas = [d.to(device) for d in datas]

    for name, layer in model.named_modules():
        print(name, layer)


    activations = extractor(model, datas, layer_types=['Conv2d', 'Linear','P4ConvZ2','P4MConvZ2'])
    print(list(activations.keys()))


    for layer, data, in activations.items():
        X = [d.reshape(d.shape[0], -1).T for d in data]
        # Get the number of features in the flattened data
        N = X[0].shape[0]
        # If N is greater than 5000, do the random projection to 5000 features
        if N > 5000:
            print("Projecting {}".format(layer))
            M = np.random.randn(5000, N)
            M /= np.sqrt(np.sum(M*M, axis=1, keepdims=True))
            X = [np.matmul(M, d) for d in X]
        activations[layer] = X


    capacities = []
    radii = []
    dimensions = []
    correlations = []

    for k, X, in activations.items():
        # Analyze each layer's activations
        a, r, d, r0, K = manifold_analysis_corr(X, 0, 300, n_reps=1)
        
        # Compute the mean values
        a = 1/np.mean(1/a)
        r = np.mean(r)
        d = np.mean(d)
        print("{} capacity: {:4f}, radius {:4f}, dimension {:4f}, correlation {:4f}".format(k, a, r, d, r0))
        
        # Store for later
        capacities.append(a)
        radii.append(r)
        dimensions.append(d)
        correlations.append(r0)

    mftm_results = {'capacity':capacities, 'radi':radii, 'dimensions':dimensions, 'correlations':correlations}
    with open(fn_mftm_results, 'wb') as f:
        pickle.dump(mftm_results, f)


    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    axes[0].plot(capacities, linewidth=5)
    axes[1].plot(radii, linewidth=5)
    axes[2].plot(dimensions, linewidth=5)
    axes[3].plot(correlations, linewidth=5)

    axes[0].set_ylabel(r'$\alpha_M$', fontsize=18)
    axes[1].set_ylabel(r'$R_M$', fontsize=18)
    axes[2].set_ylabel(r'$D_M$', fontsize=18)
    axes[3].set_ylabel(r'$\rho_{center}$', fontsize=18)

    names = list(activations.keys())
    names = [n.split('_')[1] + ' ' + n.split('_')[2] for n in names]
    for ax in axes:
        ax.set_xticks([i for i, _ in enumerate(names)])
        ax.set_xticklabels(names, rotation=90, fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    fig.savefig(fn_fig)   # save the figure to file
    plt.close(fig)

    print('DONE!')