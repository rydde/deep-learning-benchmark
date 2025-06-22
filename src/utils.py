import torch
from torchvision import datasets, transforms

def get_cifar10_loaders(batch_size=128, data_dir='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader