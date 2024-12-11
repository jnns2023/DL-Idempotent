import torch
import os
from torchvision.utils import save_image

def sample_noise_from_input(img_no, img_channels, img_size, device):
    """
    Generate noise directly in the input image space.
    NOTE: This function samples noise from the initial input space, not the latent space (black box).
    
    Args:
    - img_no (int): Number of noise samples to generate.
    - img_channels (int): Number of image channels (e.g., 3 for RGB).
    - img_size (int): Size of the image (e.g., 64 for 64x64).
    - device (torch.device): Device to generate noise on.
    
    Returns:
    - torch.Tensor: Random noise tensor of shape (img_no, img_channels, img_size, img_size).
    """
    return torch.randn(img_no, img_channels, img_size, img_size, device=device)


def save_images(images, path, filename="generated"):
    """
    Save generated images as a grid to a specified path.
    
    Args:
    - images (torch.Tensor): Batch of images to save.
    - path (str): Directory to save the images.
    - filename (str): Prefix for the saved file.
    """
    os.makedirs(path, exist_ok=True)
    save_image(images, os.path.join(path, f"{filename}.png"), nrow=8, normalize=True)


def generate_images(f, z, img_no, img_channels, img_size, device, save_path="generated_samples"):
    """Generate and save images from input-space noise using the entire model."""
    f.eval()  # Set model to evaluation mode
    with torch.no_grad():
        generated_images = f(z)
    save_images(generated_images, save_path, "generated_images")


def generate_frequency_noise(batch_data, num_generations=None):
    """
    Generate noise using frequency statistics of the real data.

    Parameters:
    batch_data (torch.Tensor): A batch of real images (B, C, H, W).

    Returns:
    torch.Tensor: Frequency-based noise with the same shape as `batch_data`.
    """
    batch_size, channels, height, width = batch_data.shape

    if num_generations is None:
        num_generations = batch_size

    # Apply FFT to real data along spatial dimensions
    fft_data = torch.fft.fft2(batch_data)

    # Calculate mean and variance of real and imaginary parts for each frequency
    real_mean = fft_data.real.mean(dim=0)
    real_std = fft_data.real.std(dim=0)
    imag_mean = fft_data.imag.mean(dim=0)
    imag_std = fft_data.imag.std(dim=0)

    # Sample `batch_size` noise samples independently for each item in the batch
    # noise_real = torch.normal(real_mean.expand(batch_size, -1, -1, -1), 
    #                           real_std.expand(batch_size, -1, -1, -1))  # Shape: (B, C, H, W)
    # noise_imag = torch.normal(imag_mean.expand(batch_size, -1, -1, -1), 
    #                           imag_std.expand(batch_size, -1, -1, -1))  # Shape: (B, C, H, W)

    noise_real = torch.zeros((num_generations, channels, height, width), device=batch_data.device)
    noise_imag = torch.zeros((num_generations, channels, height, width), device=batch_data.device)
    for i in range(num_generations):
        noise_real[i] = torch.normal(real_mean, real_std)
        noise_imag[i] = torch.normal(imag_mean, imag_std)

    # Combine real and imaginary parts
    noise_frequency = torch.complex(noise_real, noise_imag)  # Shape: (B, C, H, W)

    # Apply inverse FFT to convert frequency noise to spatial domain
    spatial_noise = torch.fft.ifft2(noise_frequency).real  # Retain only the real part

    return spatial_noise



