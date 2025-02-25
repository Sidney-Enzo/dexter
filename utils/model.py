import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, dropout_probabilit: int = 0.5):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_probabilit)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.layer2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.layer3(out)
        out = self.relu(out)
        # no actvation and not sortmax
        return out