# ---
# 01_Matrix-Multiplication.ipynb
# ---

from functools import partial

def test(a, b, compare, compare_name=None):
    if compare_name is None:
        compare_name = compare.__name__
    assert compare(a, b),\
    f"{compare_name} check failed:\n{a}\n{b}"

def test_equality(a, b):
    test(a, b, operator.eq, "Equality")

# TODO: fix this mess
def test_approximately(a, b):
    allclose = partial(torch.allclose, atol=1e-5, rtol=1e-03)
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        a = torch.tensor(a)
        b = torch.tensor(b)
    test(a, b, allclose, "Approximate Equality")

# ---
# 02_Neural-Network-Forward-Pass
# ---

from torchvision.datasets import MNIST

def stats(x):
    shape = f"Shape: {' x '.join(str(dim) for dim in x.shape)}"
    mean = f"Mean: {x.mean():.3f}"
    std = f"Std: {x.std():.3f}"
    print("\n".join([shape,mean,std]))

def normalize(x, mean, std): return (x-mean)/std

# TODO: fix this mess
def get_data():
    dataset = MNIST(root="../data/")
    x, y = dataset.data.float(), dataset.targets
    x_train, x_test = x[:50000], x[50000:]
    y_train, y_test = y[:50000], y[50000:]
    x_train = x_train.reshape(x_train.shape[0],
                              x_train.shape[1]*x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0],
                            x_test.shape[1]*x_test.shape[2])
    return x_train, y_train, x_test, y_test

# ---
# 03_Neural-Network-Backpropagation
# ---

import operator

def test_zero(x, tol=1e-3):
    test(x, tol, operator.le,
         f"Zero (less than tolerance: {tol})")

# ---
# 04_Rebuilding-PyTorch-Essentials
# ---

import torch
from torch.utils.data import DataLoader

def accuracy(input, target):
    return (torch.argmax(input, dim=-1)==target).float().mean()

class Dataset():
    def __init__(self, x, y): self.x, self.y = x, y    
    def __len__(self):        return self.x.shape[0]
    def __getitem__(self, i): return self.x[i], self.y[i]

def get_dataloaders(x_train, y_train, x_test, y_test, batch_size):
    train_ds = Dataset(x_train, y_train)
    test_ds = Dataset(x_test, y_test)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size*2, shuffle=False)
    return train_dl, test_dl

# ---
#
# ---


