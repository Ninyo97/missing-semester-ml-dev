import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

from models.mlp import MLP
from datasets import load_mnist_datasets

from sklearn.metrics import accuracy_score

# Load MNIST datasets using the refactored function
traindl, testdl, trainset, testset = load_mnist_datasets(
    batch_size_train=1, 
    batch_size_test=1, 
    shuffle_train=True, 
    shuffle_test=False
)

print(isinstance(trainset, Dataset))


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
