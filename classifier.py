import os
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

from data_loader import load_CelebA

def run_classifier():
  # select correct device
  testing = True
  if torch.cuda.is_available():
      device = torch.device('cuda')
      testing = False
  elif torch.backends.mps.is_available():
      device = torch.device('mps')
  else:
      device = torch.device('cpu')

  # define model safe directory
  if (testing):
      save_path = "test_checkpoints/"
  else:
      save_path = "checkpoints/"
  save_path += 'classifier' + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

  os.makedirs(save_path, exist_ok=True)
  hparams = {
    "learning_rate": 0.001,
    "n_epochs": 30,
    "batch_size": 512,
    "device": device,
    'testing': testing, 
    'save_path': save_path, 
    'save_interval': 5,
    'log_path': 'classifier' + datetime.now().strftime("%Y%m%d-%H%M%S")
  }

  dataloader, testloader = load_CelebA(hparams["batch_size"], num_workers=2)

  dataloaders = {
     "train": dataloader,
     "val": testloader
  }

  model = models.vgg16()  # VGG with 16 layers
  # model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # Adjust input layer
  model.avgpool = nn.AdaptiveAvgPool2d(output_size=(2, 2))
  model.classifier[0] = nn.Linear(2048, 4096)  # Input size changed to 2048
  model.classifier[-1] = nn.Linear(4096, 40)  
  model.to(device)

  opt = optim.Adam(model.parameters(), lr=hparams["learning_rate"])

  # print(model)

  print(f"Training using {device}")
  train(model, opt, dataloaders, hparams)

def train(model, opt:optim.Optimizer, dataloaders, hparams):
  if hparams['testing']:
    log_path = 'test_runs/'
  else:
    log_path = 'runs/'

  writer = SummaryWriter(log_dir=log_path + hparams['log_path'])
  hparams_text = "\n".join([f"{key}: {value}" for key, value in hparams.items()])
  writer.add_text('Hyperparameters', hparams_text)
  criterion = nn.BCEWithLogitsLoss()

  counter = {
     "train": 0,
     "val": 0
  }
  for epoch in range(hparams['n_epochs']):
    total_correct = 0
    total_samples = 0
    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()  # Set model to training mode
      else:
        model.eval()  # Set model to evaluation mode

      running_loss = 0.0

      # Iterate over data
      for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} phase"):
        inputs = inputs.to(hparams["device"])  
        labels = labels.to(hparams["device"]).float()  
        # Zero the parameter gradients
        opt.zero_grad()

        # Forward pass
        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(inputs)
          loss = criterion(outputs, labels)

          # Backward pass and optimize only in training phase
          if phase == 'train':
            loss.backward()
            opt.step()
          else:
            # Calculate predictions
            predictions = torch.sigmoid(outputs)
            binary_predictions = (predictions >= 0.5).float()

            # Count correct predictions
            correct = (binary_predictions == labels).float().sum()
            total_correct += correct.item()
            total_samples += labels.numel()

        # Accumulate loss
        running_loss += loss.item() * inputs.size(0)
        writer.add_scalar(f"Loss/{phase}_batch", loss, counter[phase])
        counter[phase] += 1

      # Compute epoch loss
      epoch_loss = running_loss / len(dataloaders[phase].dataset)
      writer.add_scalar(f"Loss/{phase}", epoch_loss, epoch+1)
      if phase == "val":
        writer.add_scalar(f"Accuracy/validation", total_correct / total_samples, epoch + 1)

      if ((epoch+1) % hparams['save_interval'] == 0):
        if (epoch+1 == hparams["n_epochs"]):
          checkpoint_path = hparams["save_path"] + "final.pth"
        else:
          checkpoint_path = hparams["save_path"] + f"epoch_{epoch + 1}.pth"
        torch.save({
          'epoch': epoch + 1,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': opt.state_dict(),
        }, checkpoint_path)
       
  return model


if __name__ == "__main__":
  run_classifier()
