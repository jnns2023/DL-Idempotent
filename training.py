import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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
        progress = tqdm(data_loader, desc=f'Epoch: {epoch + 1}/{hparams["n_epochs"]}')
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