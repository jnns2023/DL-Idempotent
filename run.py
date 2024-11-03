import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_loader import load_CelebA
from idem_net import IdemNet



def main():
    # Loss term weights
    lambda_rec = 20
    lambda_idem = 20
    lambda_tight = 2.5
    # Loss tight clamp ratio
    L_t_ratio = 1.5
    # Optimizer parameters
    alpha = 0.0001
    beta1 = 0.5  
    beta2 = 0.999
    # Training parameters
    n_epochs = 10
    batch_size = 64

    # Load data
    data_loader, test_loader = load_CelebA(batch_size)

    # Initialize f and f_copy
    f = IdemNet(image_channels=3)
    f_copy = IdemNet(image_channels=3)


    # Initialize optimizer
    opt = optim.Adam(f.parameters(), lr=alpha, betas=(beta1, beta2))

    # Train
    train(f, f_copy, lambda_idem, lambda_rec, lambda_tight, opt, data_loader, n_epochs)

    # Save model
    torch.save(f.state_dict(), 'model.pth')

    #Plot results
    # TODO
    

    # ## TEST
    # # Load model
    # f.load_state_dict(torch.load('model.pth'))

    # # Test
    # for x in data_loader:
    #     z = torch.randn_like(x)
    #     fx = f(x)
    #     fz = f(z)
    #     f_z = fz.detach()
    #     ff_z = f(f_z)
    #     f_fz = f_copy(fz)

    #     # calculate losses
    #     loss_rec = (fx - x).pow(2).mean()
    #     loss_idem = (f_fz - fz).pow(2).mean()
    #     loss_tight = -(ff_z - f_z).pow(2).mean()

    #     print(f"Rec: {loss_rec}, Idem: {loss_idem}, Tight: {loss_tight}")

    #     break



def train(f, f_copy, lambda_idem, lambda_rec, lambda_tight, opt, data_loader, n_epochs):
    for epoch in range(n_epochs):
        progress = tqdm(data_loader, desc=f'Epoch: {epoch + 1}/{n_epochs}')
        for x, labels in progress:
            z = torch.randn_like(x)

            # apply f to get all needed
            f_copy.load_state_dict(f.state_dict())
            fx = f(x)
            fz = f(z)
            f_z = fz.detach()
            ff_z = f(f_z)
            f_fz = f_copy(fz)
            
            # calculate losses
            loss_rec = (fx - x).pow(2).mean()
            loss_idem = (f_fz - fz).pow(2).mean()
            loss_tight = -(ff_z - f_z).pow(2).mean()
            
            # optimize for losses
            loss = loss_rec * lambda_rec + loss_idem * lambda_idem + loss_tight * lambda_tight
            opt.zero_grad()
            loss.backward()
            opt.step()

if __name__ == "__main__":
    main()