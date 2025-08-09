import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import numpy as np

from sklearn.metrics import accuracy_score

os.makedirs('./MNIST', exist_ok=True)

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,))
    ]
)


trainset = torchvision.datasets.MNIST(root='./MNIST/train', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./MNIST/test', train=False, download=True, transform=transform)

# Skipping transforms

print('Datasets are loaded!')
print(isinstance(trainset, Dataset))



traindl = DataLoader(dataset=trainset, batch_size=1, shuffle=True)
testdl = DataLoader(dataset=testset, batch_size=1, shuffle=False)


class MLP(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.fc1 = nn.Linear(in_features=self.in_features, out_features=50)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(in_features=50, out_features=out_features)

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



model = MLP(in_features=784, out_features=10)



def train(m, dl, max_epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3)
    for epoch in range(max_epochs):
        epoch_loss = []
        for batch in tqdm(dl):
            optimizer.zero_grad()
            x, y = batch
            preds = m(x)
            loss = criterion(preds, y)
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f"Epoch Loss: {torch.mean(torch.Tensor(epoch_loss))}")

    return m


def test(m, dl):
    predictions = []
    y_true = []
    for batch in tqdm(dl):
        x, y = batch
        preds = torch.argmax(m(x), dim=1)
        predictions.append(preds)
        y_true.append(y)

    return accuracy_score(y_true=np.array(y_true), y_pred=np.array(predictions))
    

trained_model = train(model, traindl, 5)
acc = test(trained_model, testdl)

print(f'Test Accuracy after training 5 epochs: {acc}')
