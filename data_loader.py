import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

def load_MNIST(batch_size=1):
  # TODO discuss this because this does not quite fit how they describe it in the paper
  transform = transforms.Compose([
    # transforms.Pad(padding=2),
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

  train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
  test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


  return train_loader, test_loader

def load_CelebA(batch_size=1, num_workers=2):
  # TODO discuss this because it is not mentioned in the paper
  transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize((64,64)),
    transforms.ToTensor()
  ])

  training_data = datasets.CelebA(
    root='data/',
    split='train',
    # target_type='attr',
    transform=transform,
    download=True
  )

  test_data = datasets.CelebA(
    root='data/',
    split='valid',
    # target_type='attr',
    transform=transform,
    download=True
  )

  train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
  test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

  return train_loader, test_loader