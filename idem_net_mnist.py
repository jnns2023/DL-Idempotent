import torch
from torch import nn
import torch.nn.functional as F

def init_weights(m):
  if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
    nn.init.normal_(m.weight, mean=0.0, std=0.02)
    nn.init.constant_(m.bias, 0.0)

class Encoder(nn.Module):
  def __init__(self, in_channels=1, relu_slope=0.2, momentum=0.1, layer_norm=False):
    super(Encoder, self).__init__()

    self.relu_slope = relu_slope

    self.conv1 = nn.Conv2d(in_channels, out_channels=64, kernel_size=3, stride=2, padding=1)  # out: 64 x 14 x 14

    self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1) # out: 128 x 7 x 7
    if layer_norm:
      self.bn2 = nn.LayerNorm([128, 7, 7])
    else:
      self.bn2 = nn.BatchNorm2d(128, momentum=momentum)

    self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1) # out: 256 x 4 x 4
    if layer_norm:
      self.bn3 = nn.LayerNorm([256, 4, 4])
    else:
      self.bn3 = nn.BatchNorm2d(256, momentum=momentum)

    self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=0) # out: 512 x 1 x 1


  def forward(self, x):
    x = F.leaky_relu(self.conv1(x), self.relu_slope)

    x = F.leaky_relu(self.bn2(self.conv2(x)), self.relu_slope)

    x = F.leaky_relu(self.bn3(self.conv3(x)), self.relu_slope)

    x = self.conv4(x)

    return x

class Decoder(nn.Module):
  def __init__(self, out_channels=1, momentum=0.1, layer_norm=False):
    super(Decoder, self).__init__()

    self.tconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=1, padding=0) # out: 256 x 4 x 4
    if layer_norm:
      self.bn1 = nn.LayerNorm([256, 4, 4])
    else:
      self.bn1 = nn.BatchNorm2d(256, momentum=momentum)

    self.tconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1) # out: 128 x 7 x 7
    if layer_norm:
      self.bn2 = nn.LayerNorm([128, 7, 7])
    else:
      self.bn2 = nn.BatchNorm2d(128, momentum=momentum)
    

    self.tconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1) # out: 64 x 14 x 14
    if layer_norm:
      self.bn3 = nn.LayerNorm([64, 14, 14])
    else:
      self.bn3 = nn.BatchNorm2d(64, momentum=momentum)

    self.tconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=out_channels, kernel_size=4, stride=2, padding=1) # out: 1 x 28 x 28

  def forward(self, x):
    x = F.relu(self.bn1(self.tconv1(x)))

    x = F.relu(self.bn2(self.tconv2(x)))

    x = F.relu(self.bn3(self.tconv3(x)))

    x = F.tanh(self.tconv4(x))

    return x



class IdemNetMnist (nn.Module):
  def __init__(self, image_channels=1, momentum=0.1, layer_norm=False) -> None:
    super(IdemNetMnist, self).__init__()

    # define layers
    self.encoder = Encoder(image_channels, momentum=momentum, layer_norm=layer_norm)
    self.encoder.apply(init_weights)  # set weights to papers initialization

    self.decoder = Decoder(image_channels, momentum=momentum, layer_norm=layer_norm)
    self.decoder.apply(init_weights)

  def forward(self, x):
    x = self.encoder(x)
    
    x = self.decoder(x)

    return x