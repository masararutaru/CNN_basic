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
#trainloaderの一枚の画像の形状は[ 3, 32, 32]


n_output = 10
n_hidden = 128

print("model making...")
class CNN(nn.Module):
    def __init__(self, n_outputs, n_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d((2,2))
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(32*14*14,n_hidden)
        self.l2 = nn.Linear(n_hidden, n_outputs)

        self.features = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.maxpool)

        self.classifier = nn.Sequential(
            self.l1,
            self.relu,
            self.l2)

    def forward(self, x):
        x1 = self.features(x)
        x2 = self.flatten(x1)
        x3 = self.classifier(x2)
        return x3


net = CNN(n_output, n_hidden)

criterion = nn.CrossEntropyLoss()

lr = 0.01

optimizer = optim.SGD(net.parameters(), lr = lr)

n_epochs = 10



