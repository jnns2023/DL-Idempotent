# Latent Space Manipulations. We demonstrate IGN has a consistent latent space by performing
# manipulations, similarly as shown for GANs (Radford et al., 2015). Latent space interpolation
# videos can be found in the supplementary material. We sample several random noises, take linear
# interpolation between them and apply f. In The videos left to right: z,f(z),f(f(z)),f(f(f(z))).
# Fig. 6 shows latent space arithmetics. Formally, we consider three inputs zpositive,znegative and z,
# such that f(zpositive) has a specific image property that f(znegative) and f(z) do not have (e.g. the
# faces in the two former images have glasses, while the latter does not have them). The result of
# f(zpositive−znegative) + z) is an edited version of f(z) that has the property.

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random

import numpy as np
from PIL import Image
import cv2


attribute_zpositive = 'Eyeglasses'


class CelebAWithAttributes(Dataset):
    def __init__(self, data, attr_dict):
        self.data = data
        self.attr_dict = attr_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_images_with_attribute(self, attribute, num_samples):
        if attribute not in self.attr_dict:
            raise ValueError(f"Attribute '{attribute}' not found in dataset attributes.")
        
        # Randomly select a specified number of samples with the attribute
        selected_indices = random.sample(self.attr_dict[attribute], min(num_samples, len(self.attr_dict[attribute])))
        return [self.data[i][0] for i in selected_indices]  # Returns images only
    
    def get_images_without_attribute(self, attribute, num_samples):
        if attribute not in self.attr_dict:
            raise ValueError(f"Attribute '{attribute}' not found in dataset attributes.")

        # Randomly select a specified number of samples without the attribute
        selected_indices = random.sample([i for i in range(len(self.data)) if i not in self.attr_dict[attribute]], min(num_samples, len(self.data) - len(self.attr_dict[attribute])))
        return [self.data[i][0] for i in selected_indices]


# load data
def load_CelebA(batch_size=1):
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Load the training dataset with attributes
    training_data = datasets.CelebA(
        root='data/',
        split='train',
        target_type='attr',
        transform=transform,
        download=True
    )

    # Build an attribute dictionary
    attr_dict = {attr: [] for attr in training_data.attr_names}
    for i, (_, attr) in enumerate(training_data):
        for j, has_attr in enumerate(attr):
            if has_attr == 1:
                attr_dict[training_data.attr_names[j]].append(i)

    # Use the custom dataset class with the attribute dictionary
    train_data = CelebAWithAttributes(training_data, attr_dict)
    #test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_data
    
# mean of X images with a given attribute
def mean_of_attribute(data, num_samples = 3, attribute = 'Eyeglasses'):
    
    return torch.stack(data).mean(dim=0)

def save_image(model, img, path):
    img = img.unsqueeze(0)
    img = img.to(device)
    img = f(img)
    img = img.squeeze(0)
    img = img.cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(path)

def main():
    # select correct device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # no need for training, since this is already done

    # Load data
    train_data = load_CelebA()
    #test_imgs = next(iter(test_loader))[0].to(device)

    # Load model
    f = torch.load('model.pth').to(device)
    f.eval()

    # Get images with and without the attribute
    with_attr = train_data.get_images_with_attribute(attribute_zpositive, 3)
    without_attr = train_data.get_images_without_attribute(attribute_zpositive, 3)

    # Calculate the mean of the images with the attribute
    mean_with_attr = mean_of_attribute(with_attr)
    mean_without_attr = mean_of_attribute(without_attr)

    # Calculate the difference between the means
    diff = mean_with_attr - mean_without_attr

    # Generate image with diff
    f_diff = f(diff)

    # Select random image without the attribute
    random_img = train_data.get_images_without_attribute(attribute_zpositive, 1)[0]

    # Add f_diff to the images without the attribute
    edited_img = f_diff + random_img

    # Save the images
    save_image(mean_with_attr, 'mean_with_attr.png')
    save_image(mean_without_attr, 'mean_without_attr.png')
    save_image(f_diff, 'f_diff.png')
    save_image(random_img, 'random_img.png')
    save_image(edited_img, 'edited_img.png')

    
    

    # # Generate a video of latent space interpolation
    # z1 = torch.randn_like(random_img)
    # z2 = torch.randn_like(random_img)
    # z3 = torch.randn_like(random_img)
    # z4 = torch.randn_like(random_img)
    # z5 = torch.randn_like(random_img)
    # z6 = torch.randn_like(random_img)
    # z7 = torch.randn_like(random_img)
    # z8 = torch.randn_like(random_img)
    # z9 = torch.randn_like(random_img)
    # z10 = torch.randn_like(random_img)

    # z_list = [z1, z2, z3, z4, z5, z6, z7, z8, z9, z10]
    # f_z_list = [f(z) for z in z_list]

    # # Generate the video
    # video = cv2.VideoWriter('latent_space_interpolation.avi', cv2.VideoWriter_fourcc(*'XVID'), 1, (64, 64))
    # for i in range(10):
    #     img = f_z_list[i].squeeze(0).cpu().detach().numpy()
    #     img = np.transpose(img, (1, 2, 0))
    #     img = (img * 255).astype(np.uint8)
    #     video.write(img)
    # video.release()





    

if __name__ == "__main__":
    main()