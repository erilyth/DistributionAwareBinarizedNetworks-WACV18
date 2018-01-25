import torch
import torch.nn as nn
import torch.nn.functional as F

from newlayers.BinActiveZ import Active

class Net(nn.Module):
    def __init__(self, nClasses):
        super(Net, self).__init__()

        self.nClasses = nClasses
        self.relu = nn.ReLU()
        self.hardtanh = nn.Hardtanh()
        self.drop = nn.Dropout2d(0.4)

        self.bn1 = nn.BatchNorm2d(128)
        self.conv1 = nn.Conv2d(3, 128, kernel_size = 3, stride = 1, padding = 1)
        
        self.conv2 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.activ2 = Active()

        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding= 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.activ3 = Active()
        
        self.conv4 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding= 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.activ4 = Active()

        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)

        self.conv5 = nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1)
        self.bn5 = nn.BatchNorm2d(256)
        self.activ5 = Active()
        
        self.conv6 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)
        self.bn6 = nn.BatchNorm2d(512)
        self.activ6 = Active()

        self.maxpool3 = nn.MaxPool2d(kernel_size = 2)

        self.conv7 = nn.Conv2d(512, 1024, kernel_size = 4, stride = 1)
        self.bn7 = nn.BatchNorm2d(512)
        self.activ7 = Active()
        
        self.conv8 = nn.Conv2d(1024, nClasses, kernel_size = 1, stride = 1,  padding = 0)


    def forward(self, x):

        #print(x.size())
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(self.relu(self.conv2(self.activ2(self.bn2(x)))))

        x = self.relu(self.conv3(self.activ3(self.bn3(x))))
        x = self.maxpool2(self.relu(self.conv4(self.activ4(self.bn4(x)))))
        x = self.drop(x)
        #print(x.size())

        x = self.relu(self.conv5(self.activ5(self.bn5(x))))
        x = self.maxpool3(self.relu(self.conv6(self.activ6(self.bn6(x)))))
        x = self.drop(x)
        #print(x.size())
        x = self.relu(self.conv7(self.activ7(self.bn7(x))))
        #x = self.conv8(self.bn8(x))
        x = self.conv8(x)
        #print(x)
        x = x.view(-1, self.nClasses)

        return F.log_softmax(x)
