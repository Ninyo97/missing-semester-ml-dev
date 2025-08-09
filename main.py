import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import argparse

from models.mlp import MLP
from datasets import load_mnist_datasets

from sklearn.metrics import accuracy_score


def parse_arguments():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description='Train MLP on MNIST dataset')
    
    parser.add_argument('--batch-size-train', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--batch-size-test', type=int, default=1000,
                        help='Batch size for testing (default: 1000)')
    
    parser.add_argument('--hidden-features', type=int, default=128,
                        help='Number of hidden features in MLP (default: 128)')
    
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs (default: 5)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 0.001)')
    
    return parser.parse_args()

def train(m, dl, max_epochs, lr, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=m.parameters(), lr=lr)
    
    m.to(device)
    
    for epoch in range(max_epochs):
        epoch_loss = []
        for batch in tqdm(dl, desc=f'Epoch {epoch+1}/{max_epochs}'):
            optimizer.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            preds = m(x)
            loss = criterion(preds, y)
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} Loss: {torch.mean(torch.Tensor(epoch_loss)):.4f}")

    return m


def test(m, dl, device):
    predictions = []
    y_true = []
    
    m.eval()
    with torch.no_grad():
        for batch in tqdm(dl, desc='Testing'):
            x, y = batch
            x, y = x.to(device), y.to(device)
            preds = torch.argmax(m(x), dim=1)
            predictions.extend(preds.cpu().numpy())
            y_true.extend(y.cpu().numpy())

    return accuracy_score(y_true=np.array(y_true), y_pred=np.array(predictions))


def main():
    args = parse_arguments()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    np.random.seed(42)
    
    print(f"Configuration: {args}")
    
    traindl, testdl, trainset, testset = load_mnist_datasets(
        batch_size_train=args.batch_size_train, 
        batch_size_test=args.batch_size_test 
        )
    
    
    model = MLP(in_features=784, hidden_features=args.hidden_features, out_features=10)
    
    trained_model = train(model, traindl, args.epochs, args.lr, device)
    acc = test(trained_model, testdl, device)
    
    print(f'Test Accuracy after training {args.epochs} epochs: {acc:.4f}')


if __name__ == '__main__':
    main()
