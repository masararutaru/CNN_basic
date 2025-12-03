#docker compose exec app bash >
# python -i src/train.py >
# 書いたファイルを実行した状態から対話モードでdataの形やモデルについて確認できる
#抜けたくなったらcontrol + dを押す(2回押したら対話モードから一気に通常のターミナルまで戻る)

#学習中に Ctrl + c を1回押す
# 普通ならそこでプログラムが強制終了して終わるが、-i オプションがついている場合は中断されたその瞬間の状態で対話モード（>>>）に入れる
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


