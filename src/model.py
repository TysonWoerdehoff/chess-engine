import torch
import torch.nn as nn

class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(17, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.flatten1 = nn.Flatten()
        self.linear1 = nn.Linear(4096, 256)
        self.relu4 = nn.ReLU()
        self.linear2 = nn.Linear(256, 1)
        self.tanh =  nn.Tanh()
        
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.flatten1(x)
        x = self.linear1(x)
        x = self.relu4(x)
        x = self.linear2(x)
        x = self.tanh(x)
        return x
        
    
if __name__ == "__main__":
    model = ChessNet()
    print(model)
    n_params = sum(p.numel() for p in model.parameters())
    print("total parameters:", n_params)

    dummy = torch.randn(4, 17, 8, 8)
    out = model(dummy)
    print("output shape:", out.shape)
    print("output range:", out.min().item(), out.max().item())