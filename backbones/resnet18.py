import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

class ResidualBlock(nn.Module):    
    def __init__(self, inchannel, outchannel, stride=1):        
        super(ResidualBlock, self).__init__()        
        self.left = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(inplace=True),
                nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(outchannel)
            )        
        self.shortcut = nn.Sequential()        
        if stride != 1 or inchannel != outchannel:
                self.shortcut = nn.Sequential(
        
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
        
                nn.BatchNorm2d(outchannel)
        
            )
            
    def forward(self, x):        
        out = self.left(x)        
        out = out + self.shortcut(x)        
        out = F.relu(out)                
        return out    
        
class ResNet(nn.Module):
    def __init__(self, ResidualBlock, layers, input_channels, num_classes = 10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)        
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)        
        self.fc = nn.Linear(512, num_classes)
        
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
            return nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = F.softmax(out, dim=1)  # Apply softmax activation
        return out    
    
    
    def predict(self,model, testloader, device):
        criterion = torch.nn.CrossEntropyLoss()
        loss = 0.0
        model.eval()
        model.to(device)
        with torch.no_grad():        
            for data in tqdm(testloader):           
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                loss += criterion(outputs, labels).item()
    
    

# Create an instance of the ResNet18 network with customizable number of classes
def ResNet18(num_classes, input_channels):
    return ResNet(ResidualBlock, [2,2,2,2], input_channels, num_classes)
