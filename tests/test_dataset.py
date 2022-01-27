from ambiguous.dataset.dataset import *

def test_aMNIST_dataset():
    root = 'data/'
    trainset = aMNIST(root=root, download=True, train=True, transform=None)
    testset = aMNIST(root=root, download=True, train=False, transform=None)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)
    x, t = next(iter(trainloader))
    assert x.shape == (64, 1, 28, 28)

def test_aEMNIST_fly_dataset():
    batch_size=64
    dataset=EMNIST_fly('/share/datasets',blend=0.5)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    x,t = next(iter(data_loader))
    assert (x.shape[0]==batch_size and t.shape[0] == batch_size)

def test_aMNIST_fly_dataset():
    batch_size=64
    dataset=MNIST_fly('/share/datasets',blend=0.5)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    x,t = next(iter(data_loader))
    assert (x.shape[0]==batch_size and t.shape[0] == batch_size)
