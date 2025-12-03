print("importing libraries...")
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_cifar10_loaders

print("setting device...")
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f"Using device : {device}")

print("loading dataset...")
trainloader, testloader = get_cifar10_loaders(batch_size = 64)
#画像の形状はtorch.Size([64, 3, 32, 32])


