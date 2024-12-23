import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from gen_utils import *
from perceptual_loss import PerceptualLoss

def train(f, f_copy, opt, data_loader, hparams, device=torch.device('cpu')):
    print(f"Training using {device}")
    if hparams['testing']:
        log_path = 'test_runs/'
    else:
        log_path = 'runs/'

    if 'perceptual_loss' in hparams.keys() and hparams['perceptual_loss']:
        perceptual_loss = PerceptualLoss(pretrained_model=hparams['perceptual_model'], device=device)
        # hparams['lambda_rec'] *= 0.1  # set the multiplicative factor for the perceptual loss to be lower

    test_imgs, _ = next(iter(data_loader))
    test_imgs = test_imgs[0:9].to(device)

    writer = SummaryWriter(log_dir=log_path + hparams['log_path'])
    hparams_text = "\n".join([f"{key}: {value}" for key, value in hparams.items()])
    writer.add_text('Hyperparameters', hparams_text)
    writer.add_images('Reconstruction', test_imgs)

    batch_count = 0
    
    z_gen = torch.randn_like(test_imgs) # Batch of noise to generate images from

    # Training loop
    for epoch in range(hparams['n_epochs']):
        progress = tqdm(data_loader, ascii=True, dynamic_ncols=True, desc=f'Epoch: {epoch + 1}/{hparams["n_epochs"]}')
        for x, labels in progress:
            x = x.to(device)
            z = torch.randn_like(x)

            # Forward pass
            f_copy.load_state_dict(f.state_dict())
            fx = f(x)
            fz = f(z)
            f_z = fz.detach()
            ff_z = f(f_z)
            f_fz = f_copy(fz)
            
            # calculate losses
            if 'perceptual_loss' in hparams.keys() and hparams['perceptual_loss']:
                loss_rec = perceptual_loss(fx, x)
            else:
                loss_rec = (fx - x).pow(2).mean()


            loss_idem = (f_fz - fz).pow(2).mean()

            # Define the constant alpha >= 1
            a = 1.5 # NOTE: might need change if the loss_tight is too high
            # Compute the original tight loss
            loss_tight_unclamped = -(fz - ff_z).pow(2).mean()
            # Clamp loss_tight using tanh
            loss_tight = torch.tanh(loss_tight_unclamped / (a * loss_rec)) * (a * loss_rec)

            if 'perceptual_loss' in hparams.keys() and hparams['perceptual_loss']:
                writer.add_scalar("Loss/loss_percept", loss_rec, batch_count)
            else:
                writer.add_scalar("Loss/loss_rec", loss_rec, batch_count)
            writer.add_scalar("Loss/loss_idem", loss_idem, batch_count)
            writer.add_scalar("Loss/loss_tight", loss_tight, batch_count)
            
            # optimize for losses
            loss = loss_rec * hparams['lambda_rec'] + loss_idem * hparams['lambda_idem'] + loss_tight * hparams['lambda_tight']
            writer.add_scalar("Loss/total", loss, batch_count)

            opt.zero_grad()
            loss.backward()

            clip_grad_norm_(f.parameters(), max_norm=10.0)

            parameters = [p for p in f.parameters() if p.grad is not None and p.requires_grad]
            if len(parameters) == 0:
                writer.add_scalar("Grad_Norm", 0, batch_count)
            else:
                device = parameters[0].grad.device
                total_norm = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters]),
                    2).item()
                writer.add_scalar("Grad_Norm", total_norm, batch_count)

            opt.step()
            batch_count += 1
               
        # Log reconstruction
        writer.add_images('Reconstruction', f(test_imgs), epoch+1)

        # Generate images
        img_no = 9 # Number of images to generate
        writer.add_images('Generation', f(z_gen), epoch+1)

        for name, module in f.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                writer.add_scalar(f"XBatchRunMean_mean/{name}", module.running_mean.mean(), epoch+1)
                writer.add_scalar(f"XBatchRunMean_var/{name}", module.running_mean.std(dim=None), epoch+1)
                writer.add_scalar(f"XBatchRunVar_mean/{name}", module.running_var.mean(), epoch+1)
                writer.add_scalar(f"XBatchRunVar_var/{name}", module.running_var.std(dim=None), epoch+1)


        # Save model
        if (epoch % hparams['save_interval'] == 0):
            checkpoint_path = hparams["save_path"] + f"epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': f.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
            }, checkpoint_path)



    writer.flush()
    writer.close()