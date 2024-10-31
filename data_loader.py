import torch 
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def load_MNIST():
  # TODO discuss this because this does not quite fit how they describe it in the paper
  transform = transforms.Compose([
    transforms.Pad(padding=2),
    transforms.ToTensor()
  ])

  training_data = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=transform
  )

  test_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=transform
  )

  return training_data, test_data