from ambiguous.dataset.dataset import *

def test_EMNIST_dataset():
    batch_size=64
    dataset=EMNIST_fly('/share/datasets',blend=0.5)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    x,t = next(iter(data_loader))
    assert (x.shape[0]==batch_size and t.shape[0] == batch_size)

def test_MNIST_dataset():
    batch_size=64
    dataset=MNIST_fly('/share/datasets',blend=0.5)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    x,t = next(iter(data_loader))
    assert (x.shape[0]==batch_size and t.shape[0] == batch_size)
