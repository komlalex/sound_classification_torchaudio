import torch 
from torch import nn 
from torchsummary import summary 

device = "cuda" if torch.cuda.is_available() else "cpu"
class CNNNetwork(nn.Module): 
    def __init__(self):
        super().__init__()  
        # 4 conv block / flatten / linear / softmax 
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, 
                      out_channels=16, 
                      kernel_size=3, 
                      stride=1, 
                      padding=2), 
                      nn.ReLU(), 
                      nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, 
                      out_channels=32, 
                      kernel_size=3, 
                      stride=1, 
                      padding=2), 
                      nn.ReLU(), 
                      nn.MaxPool2d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, 
                      out_channels=64, 
                      kernel_size=3, 
                      stride=1, 
                      padding=2), 
                      nn.ReLU(), 
                      nn.MaxPool2d(kernel_size=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, 
                      out_channels=128, 
                      kernel_size=3, 
                      stride=1, 
                      padding=2), 
                      nn.ReLU(), 
                      nn.MaxPool2d(kernel_size=2)
        )

        self.flatten = nn.Flatten() 

        self.linear = nn.Linear(128*5*4, 10) 
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        out = self.conv1(x) 
        out = self.conv2(out) 
        out = self.conv3(out) 
        out = self.conv4(out) 
        out = self.flatten(out) 
        logits = self.linear(out) 
        predictions = self.softmax(logits) 
        return predictions 

if __name__ == "__main__": 
    cnn = CNNNetwork().to(device)
    summary(cnn, (1, 64, 44))  


