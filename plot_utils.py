from matplotlib import pyplot as plt

def plot_generation(inputs, model, num_images=5, num_applications=1):
  """
  Plots input and output image pairs side by side.
  
  Parameters:
  - inputs: Batch of input images, expected shape (batch_size, height, width).
  - outputs: Batch of output images, expected shape (num_applications, batch_size, height, width).
  - num_images: Number of image pairs to display (default is 5).
  """
  # Limit the number of images to the smaller of num_images or batch size
  num_images = min(num_images, len(inputs))

  images = [inputs]
  for _ in range(num_applications):
    images.append(model(images[-1]))
  
  cols = len(images)
  plt.figure(figsize=(cols * 2, num_images * 2))
  for row in range(num_images):
    for col in range(cols):
    # Plot input image
      plt.subplot(num_images, cols, cols * row + col + 1)
      if inputs.shape[1] == 3:
        _show_rgb(images[col][row])
      else:
        _show_gray(images[col][row])
      
      if col == 0:
        plt.title("Input")
      else:
        plt.title(f"f^{col}")
      plt.axis('off')
  
  plt.tight_layout()
  plt.show()


def _show_rgb(image):
  plt.imshow(image.squeeze().permute(1,2,0).detach().numpy())

def _show_gray(image):
  plt.imshow(image.squeeze().detach().numpy(), cmap="gray")