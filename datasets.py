import torchvision
from torch.utils.data import DataLoader, random_split
import os


def load_mnist_datasets(batch_size_train=1, batch_size_test=1, shuffle_train=True, shuffle_test=False):
    os.makedirs('./MNIST', exist_ok=True)
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    
    trainset_full = torchvision.datasets.MNIST(
        root='./MNIST/train', 
        train=True, 
        download=True, 
        transform=transform
    )
    train_size = int(0.8 * len(trainset_full))
    val_size = len(trainset_full) - train_size
    trainset, valset = random_split(trainset_full, [train_size, val_size])

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
    valdl = DataLoader(
        dataset=valset,
        batch_size=batch_size_train,
        shuffle=shuffle_train
    )
    testdl = DataLoader(
        dataset=testset, 
        batch_size=batch_size_test, 
        shuffle=shuffle_test
    )
    
    print('Datasets are loaded!')
    
    return traindl, valdl, testdl, trainset, valset, testset
