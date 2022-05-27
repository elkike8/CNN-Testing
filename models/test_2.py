# =============================================================================
# imports
# =============================================================================
from inception_pytorch_v2 import *

num_classes = 14

a = conv_block(in_channels = 1, out_channels = 64, kernel_size = 7, stride = 2, padding = 3)
b = nn.MaxPool2d(kernel_size=3, stride = 2, padding = 1)

c = conv_block(64, 
                        192,
                        kernel_size = 3,
                        stride = 1,
                        padding = 1)
d = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

E = inception_block(in_channels = 192, 
                                   out_1x1 = 64, 
                                   red_3x3 = 96, 
                                   out_3x3 = 128, 
                                   red_5x5 = 16, 
                                   out_5x5 = 32, 
                                   out_1x1pool = 32)
f = inception_block(256, 128, 182, 192, 32, 96, 64)

g = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

h = inception_block(480, 192, 96, 208, 16, 48, 64)
i = inception_block(512, 160, 112, 224, 24, 64, 64)
j = inception_block(512, 128, 128, 256, 24, 64, 64)
k = inception_block(512, 112, 144, 288, 32, 64, 64)
l = inception_block(528, 256, 160, 320, 32, 128, 128)

m = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

n = inception_block(832, 256, 160, 320, 32, 128, 128)
o = inception_block(832, 384, 192, 384, 48, 128, 128)

  
p = nn.AvgPool2d(kernel_size = 7, stride = 1, padding = 0)

  
q = nn.Dropout2d(p = 0.4)
r = nn.Linear(1024, num_classes)
S = nn.Softmax(dim = 1)

z = nn.AdaptiveMaxPool2d((56,56))

x = torch.randn(1,1,24,24)

test_l1 = a(x)
test_l2 = b(test_l1)
test_l3 = c(test_l2)
test_l4 = z(test_l3)
