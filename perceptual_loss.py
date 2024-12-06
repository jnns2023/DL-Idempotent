import torch
import torch.nn as nn
import torchvision.models as models

# layers previously was 3, 8, 15
class PerceptualLoss(nn.Module):
  def __init__(self, pretrained_model='vgg16', layers=[3], device=torch.device('cpu')):
    super().__init__()
    if pretrained_model == 'vgg16':
      model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
    elif pretrained_model == 'vgg16_celeba':
      checkpoint_path = "checkpoints/classifier20241206-094306/epoch_15.pth"
      model = models.vgg16()  # VGG with 16 layers
      model.avgpool = nn.AdaptiveAvgPool2d(output_size=(2, 2))
      model.classifier[0] = nn.Linear(2048, 4096)  # Input size changed to 2048
      model.classifier[-1] = nn.Linear(4096, 40)

      state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)

      model.load_state_dict(state_dict["model_state_dict"])
      model.eval()
      model = model.features
    else:
      raise ValueError("The given model is not supported")
    
    self.device = device
    self.selected_layers = layers
    self.slices = nn.ModuleList()
    prev_index = 0
    
    # Create submodules for feature extraction
    for i in layers:
      self.slices.append(nn.Sequential(*[model[j] for j in range(prev_index, i + 1)]).to(self.device))
      prev_index = i + 1
    
    # Set requires_grad to False for the VGG layers
    for param in self.parameters():
      param.requires_grad = False

  def forward(self, x, target):
    loss = 0.0
    x_features = x.clone().to(self.device)
    target_features = target.clone().to(self.device)
    for slice in self.slices:
      x_features = slice(x_features)
      target_features = slice(target_features)
      loss += nn.functional.mse_loss(x_features, target_features)
    return loss
    
