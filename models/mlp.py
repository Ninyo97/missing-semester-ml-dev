import torch
import torch.nn as nn

class MLP(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.fc1 = nn.Linear(in_features=self.in_features, out_features=hidden_features)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=out_features)

    def forward(self, x):
        '''
        x.shape = (n, 1, 28, 28)
        out.shape = (n, 1)
        '''
        assert ((x.shape[2] == 28) and (x.shape[3] == 28), f"Input shape (n, 1, {x.shape[2]}, {x.shape[3]}) does not match expected shape (n, 1, 28, 28)")

        x = torch.flatten(x, start_dim=1, end_dim=-1)
        
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        return out

