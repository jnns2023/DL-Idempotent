import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_loader import load_MNIST
from idem_net_mnist import IdemNetMnist

from training import train



def main(testing=True):
    # select correct device
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
    save_path += 'mnist' + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

    os.makedirs(save_path, exist_ok=True)


    hparams = {
        'lambda_rec': 20,
        'lambda_idem': 20,
        'lambda_tight': 2.5,
        'L_t_ratio': 1.5,
        'alpha': 0.0001,
        'beta1': 0.5,
        'beta2': 0.999,
        'n_epochs': 10,
        'batch_size': 256,
        'testing': testing, 
        'save_path': save_path, 
        'save_interval': 5,
        'log_path': 'mnist' + datetime.now().strftime("%Y%m%d-%H%M%S")
    }

    # Load data
    data_loader, test_loader = load_MNIST(hparams['batch_size'])

    # Initialize f and f_copy
    f = IdemNetMnist().to(device)
    f_copy = IdemNetMnist().to(device)

    # Initialize optimizer
    opt = optim.Adam(f.parameters(), lr=hparams['alpha'], betas=(hparams['beta1'], hparams['beta2']))

    # Initialize logging

    # Train
    train(f, f_copy, opt, data_loader, hparams, device)

    # Save model
    torch.save(f.state_dict(), save_path + "_final.pth")

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


if __name__ == "__main__":
    main()