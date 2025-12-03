import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_cifar10_loaders

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f"Using device : {decice}")

trainloader, testloader = get_cifar10_loaders(batch_size = 64)

