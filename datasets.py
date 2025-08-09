import torchvision
from torch.utils.data import DataLoader
import os


def load_mnist_datasets(batch_size_train=1, batch_size_test=1, shuffle_train=True, shuffle_test=False):
    os.makedirs('./MNIST', exist_ok=True)
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    
    trainset = torchvision.datasets.MNIST(
        root='./MNIST/train', 
        train=True, 
        download=True, 
        transform=transform
    )
    testset = torchvision.datasets.MNIST(
        root='./MNIST/test', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    traindl = DataLoader(
        dataset=trainset, 
        batch_size=batch_size_train, 
        shuffle=shuffle_train
    )
    testdl = DataLoader(
        dataset=testset, 
        batch_size=batch_size_test, 
        shuffle=shuffle_test
    )
    
    print('Datasets are loaded!')
    
    return traindl, testdl, trainset, testset
