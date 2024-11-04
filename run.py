import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_loader import load_CelebA
from idem_net import IdemNet



def main(testing=False):
    # select correct device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # define model safe directory
    if (testing):
        save_path = "test_checkpoints/"
    else:
        save_path = "checkpoints/"
    save_path += 'celeba' + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

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
        'save_interval': 5
    }

    # Load data
    data_loader, test_loader = load_CelebA(hparams['batch_size'])

    # Initialize f and f_copy
    f = IdemNet(image_channels=3).to(device)
    f_copy = IdemNet(image_channels=3).to(device)




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



def train(f, f_copy, opt, data_loader, hparams, device=torch.device('cpu')):
    print(f"Training using {device}")
    if hparams['testing']:
        log_path = 'test_runs/'
    else:
        log_path = 'runs/'

    test_imgs, _ = next(iter(data_loader))
    test_imgs = test_imgs[0:9].to(device)

    writer = SummaryWriter(log_dir=log_path + 'celeba' + datetime.now().strftime("%Y%m%d-%H%M%S"))
    hparams_text = "\n".join([f"{key}: {value}" for key, value in hparams.items()])
    writer.add_text('Hyperparameters', hparams_text)
    writer.add_images('Generation', test_imgs)

    batch_count = 0
    for epoch in range(hparams['n_epochs']):
        progress = tqdm(data_loader, desc=f'Epoch: {epoch + 1}/{hparams['n_epochs']}')
        for x, labels in progress:
            x = x.to(device)
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

            writer.add_scalar("Loss/loss_rec", loss_rec, batch_count)
            writer.add_scalar("Loss/loss_idem", loss_idem, batch_count)
            writer.add_scalar("Loss/loss_tight", loss_tight, batch_count)
            
            # optimize for losses
            loss = loss_rec * hparams['lambda_rec'] + loss_idem * hparams['lambda_idem'] + loss_tight * hparams['lambda_tight']
            writer.add_scalar("Loss/total", loss, batch_count)

            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_count += 1
        
        writer.add_images('Generation', f(test_imgs), epoch+1)
        if (epoch % hparams['save_interval'] == 0):
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': f.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
            }, hparams["save_path"] + f"epoch_{epoch + 1}.pth")

    writer.flush()
    writer.close()



if __name__ == "__main__":
    main()