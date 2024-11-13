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

