from torchvision.models import resnet50
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, Bottleneck


# load up ResNet50
class singleGPUResNet50(torch.nn.Module):
    # default to 10 classes because we are using CIFAR 10
    def __init__(self, num_classes = 10):
        super(singleGPUResNet50, self).__init__()
        # Load the pre-trained ResNet-50 model
        self.resnet = resnet50(pretrained=False)

        # we have to modify the last layer to match the number of classes       
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
    

# from pytorch docs: https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html
class ModelParallelResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(ModelParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=10, *args, **kwargs)

        self.seq1 = torch.nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        ).to('cuda:0')

        self.seq2 = torch.nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to('cuda:1')

        self.fc.to('cuda:1')

    def forward(self, x):
        x = self.seq2(self.seq1(x).to('cuda:1'))
        return self.fc(x.view(x.size(0), -1))
    

class SimpleParallel(nn.Module):
    def __init__(self):
        super(SimpleParallel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5).to('cuda:0')
        self.pool = nn.MaxPool2d(2, 2).to('cuda:0')
        self.conv2 = nn.Conv2d(6, 16, 5).to('cuda:0')
        self.fc1 = nn.Linear(16 * 5 * 5, 120).to('cuda:1')
        self.fc2 = nn.Linear(120, 84).to('cuda:1')
        self.fc3 = nn.Linear(84, 10).to('cuda:1')

    def __str__(self):
        return "CNN"

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5).to('cuda:1')
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x