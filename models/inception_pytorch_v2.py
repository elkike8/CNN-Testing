from inception_pytorch_v1 import googlenet
import torch
import torch.nn as nn

model = googlenet(in_channels=3, num_classes=10)

x = torch.randn(3,3,28,28)

model(x).shape
