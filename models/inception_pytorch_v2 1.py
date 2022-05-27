import torch
import torch.nn as nn


class googlenet(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 10):
        super(googlenet, self).__init__()
        
        self.conv1 = convolution_block(in_channels = in_channels,
                                       out_channels = 64,
                                       kernel_size = 7,
                                       stride = 2,
                                       padding = 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,
                                     stride = 2,
                                     padding = 1)
        
        self.conv2 = convolution_block(64,
                                       192,
                                       kernel_size = 3,
                                       stride = 1,
                                       padding = 1)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.inception3a = inception_module(in_channels = 192, 
                                           out_1x1 = 64, 
                                           red_3x3 = 96, 
                                           out_3x3 = 128, 
                                           red_5x5 = 16, 
                                           out_5x5 = 32, 
                                           out_1x1pool = 32)
        self.inception3b = inception_module(256, 128, 182, 192, 32, 96, 64)
        
        self.maxpool3 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.inception4a = inception_module(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_module(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_module(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_module(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_module(528, 256, 160, 320, 32, 128, 128)
        
        self.maxpool4 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.inception5a = inception_module(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_module(832, 384, 192, 384, 48, 128, 128)
        
        #to manage bigger images
        self.adaptative = nn.AdaptiveMaxPool2d((7,7))
    
        self.avgpool = nn.AvgPool2d(kernel_size = 7, stride = 1, padding = 0)
        
      
        self.dropout = nn.Dropout2d(p = 0.4)
        self.linear = nn.Linear(1024, num_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.adaptative(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        
        x = self.dropout(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        
        return x


class inception_module(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
       super(inception_module, self).__init__()
       
       self.branch1 = convolution_block(in_channels, out_1x1, kernel_size = (1,1))
       
       self.branch2 = nn.Sequential(
           convolution_block(in_channels, red_3x3, padding = 0, kernel_size=1),
           convolution_block(red_3x3, out_3x3, kernel_size = 3, stride = 1, padding = 1)
           )

       self.branch3 = nn.Sequential(
            convolution_block(in_channels, red_5x5, kernel_size = 1),
            convolution_block(red_5x5, out_5x5, kernel_size = 5, padding = 2)
            )
       
       self.branch4 = nn.Sequential(
           nn.MaxPool2d(kernel_size=3, stride = 1, padding = 1),
           convolution_block(in_channels, out_1x1pool, kernel_size=1)
           )

    def forward(self, x):
        # concatenate over the number of filters
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)

class convolution_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(convolution_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        # addition for Inception V2
        self.batchnorm = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):        
        return self.relu(self.batchnorm(self.conv(x)))
    
if __name__ == "__main__":
    x = torch.randn(3, 3, 224, 224)
    model = googlenet()
    print(model(x).shape)

