
from torch.nn import Module

class LambdaLayer(Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd
    
    def forward(self, x):
        return self.lambd(x)