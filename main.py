import torchvision
from torch.utils.data import Dataset
import os

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
