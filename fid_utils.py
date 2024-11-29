import warnings
import torch
import numpy as np
from itertools import islice
from torchvision.models import inception_v3, Inception_V3_Weights
import torch.nn.functional as F
from scipy.linalg import sqrtm
from tqdm import tqdm
from data_loader import load_CelebA
from idem_net_celeba import IdemNetCeleba

# TODO remove again
from torch.utils.data import DataLoader, TensorDataset


def preprocess_images(images, target_size=299):
    # Resize using interpolate (for tensors)
    images_resized = F.interpolate(images, size=(target_size, target_size), mode='bilinear', align_corners=False)
    
    # Normalize images to [-1, 1]
    images_normalized = (images_resized - 0.5) / 0.5  # Assuming the images are in range [0, 1]
    
    return images_normalized


def get_inception_model():
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
    # print(model)
    # model.Mixed_7c.register_forward_hook(_hook_fn)  # Attach the hook to the last pooling layer
    model.fc = torch.nn.Identity()  # Remove the classification head
    model.eval()
    return model

# Get the mean and covariance of the feature space (last pooling layer)
def compute_statistics(batch, running_mean, running_scatter, total_count):
    batch_count = batch.size(0)
    total_count += batch_count

    batch_mean = torch.mean(batch, dim=0)
    delta_mean = batch_mean - running_mean

    running_mean += delta_mean * (batch_count / total_count)

    batch_centered = batch - batch_mean
    batch_scatter = batch_centered.T @ batch_centered

    # Update running scatter matrix
    running_scatter += batch_scatter + torch.outer(delta_mean, delta_mean) * batch_count * (total_count - batch_count) / total_count

    return total_count

# Compute the Fréchet distance
def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = sqrtm(np.dot(sigma1, sigma2), disp=False)

    if not np.isfinite(covmean).all():
      warnings.warn(f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates")
      offset = np.eye(sigma1.shape[0]) * eps
      covmean, _ = sqrtm(np.dot((sigma1 + offset), (sigma2 + offset)))

    # Numerical stability: if covmean has imaginary values, take only the real part
    if np.iscomplexobj(covmean):
        if not np.allclose(covmean.imag, 0, atol=1e-3):
          err = np.max(np.abs(covmean.imag))
          warnings.warn(f"Covariance matrices are not positive semi-definite. Error: {err}")
        covmean = covmean.real
    return np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)

# Function to compute FID score between f_z and ff_z
def compute_fid(dataloader, generator, num_generations=50000, feature_dim=2048, device="cpu", use_precalc=True):
  inception_model = get_inception_model().to(device)

  batch_size = dataloader.batch_size

  max_batches = int(num_generations/ batch_size)

  if not use_precalc:
    progress = tqdm(islice(dataloader, max_batches), total=max_batches, desc="Target images")

    # calculate mean and variance for images
    running_mean = torch.zeros(feature_dim, device=device)
    running_scatter = torch.zeros((feature_dim, feature_dim), device=device)
    total_count = 0
    with torch.no_grad():
      for images, _ in progress:
        output = inception_model(preprocess_images(images).to(device))
        flat_output = torch.flatten(output, start_dim=1)

        # updates running statistics in place
        total_count = compute_statistics(flat_output, running_mean, running_scatter, total_count)
      
    image_mean = running_mean
    image_cov = running_scatter / (total_count - 1)

    image_mean = image_mean.cpu().detach().numpy()
    image_cov = image_cov.cpu().detach().numpy()
  else:
    print("Using precalculated statistics")

    images, _ = next(iter(dataloader))
    statistics = np.load("./data/fid_stats_celeba.npz")
    image_mean = statistics['mu']
    image_cov = statistics['sigma']


  # calculate mean and variance for generations
  running_mean = torch.zeros(feature_dim, device=device)
  running_scatter = torch.zeros((feature_dim, feature_dim), device=device)
  total_count = 0
  progress = tqdm(range(max_batches), desc="Generations")
  with torch.no_grad():
    for _ in progress:
      generated_batch = generator(torch.randn_like(images, device=device))
      output = inception_model(preprocess_images(generated_batch).to(device))
      flat_output = torch.flatten(output, start_dim=1)

      # updates running statistics in place
      total_count = compute_statistics(flat_output, running_mean, running_scatter, total_count)
    
  generation_mean = running_mean
  generation_cov = running_scatter / (total_count - 1)

  # Calculate FID
  fid_score = frechet_distance(
    image_mean,
    image_cov,
    generation_mean.cpu().detach().numpy(),
    generation_cov.cpu().detach().numpy()
  )
  return fid_score

if __name__ == "__main__":
  # usage between two random tensors
  statistics = np.load("./data/fid_stats_celeba.npz")

  batch_size = 32
  image_size = (3, 64, 64)
  random_images = torch.randn(batch_size, *image_size)  # Random image batch
  random_labels = torch.randint(0, 10, (batch_size,))  # Random labels

  # Wrap in a TensorDataset
  mock_dataset = TensorDataset(random_images, random_labels)

  # Create the DataLoader
  mock_data_loader = DataLoader(mock_dataset, batch_size=batch_size)

  data_loader, _ = load_CelebA(32)

  device = torch.device("cpu")
  if torch.cuda.is_available():
      device = torch.device('cuda')
  elif torch.backends.mps.is_available():
      device = torch.device('mps')
  else:
      device = torch.device('cpu')

  generator = IdemNetCeleba(3).to(device)
  fid_score = compute_fid(data_loader, generator, num_generations=4096, device=device)
  print(f"FID score between target images and generations: {fid_score}")