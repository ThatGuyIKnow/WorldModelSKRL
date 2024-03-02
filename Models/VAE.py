
"""
Variational encoder model, used as a visual model
for our model of the world.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
import lightning as L

class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self, img_channels, latent_size):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.fc1 = nn.Linear(latent_size, 1024)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 6, stride=2, padding=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 32, 6, stride=2, padding=2)
        self.deconv5 = nn.ConvTranspose2d(32, img_channels, 4, stride=2, padding=1)

    def forward(self, x): # pylint: disable=arguments-differ
        x = F.leaky_relu(self.fc1(x))
        x = x.view(x.size(0), -1, 2, 2)
        x = F.leaky_relu(self.deconv1(x))
        x = F.leaky_relu(self.deconv2(x))
        x = F.leaky_relu(self.deconv3(x))
        x = F.leaky_relu(self.deconv4(x))
        reconstruction = F.sigmoid(self.deconv5(x))
        return reconstruction

class Encoder(nn.Module): # pylint: disable=too-many-instance-attributes
    """ VAE encoder """
    def __init__(self, img_channels, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        #self.img_size = img_size
        self.img_channels = img_channels

        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        self.fc_mu = nn.Linear(2*2*256, latent_size)
        self.fc_logsigma = nn.Linear(2*2*256, latent_size)
    

    def forward(self, x): # pylint: disable=arguments-differ
        if len(x.shape) == 3:
            x = x.view(1, *x.shape)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)

        z = self.sample_z(mu, logsigma)

        return mu, logsigma, z
    
    def sample_z(self, mu, logsigma):
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        return eps.mul(sigma).add_(mu)


class VAE(L.LightningModule):
    """ Variational Autoencoder """
    def __init__(self, img_channels, latent_size):
        super(VAE, self).__init__()
        self.save_hyperparameters()

        self.img_channels = img_channels
        self.encoder = Encoder(img_channels, latent_size)
        self.decoder = Decoder(img_channels, latent_size)

    def forward(self, x): # pylint: disable=arguments-differ
        #x = x.view(-2, self.img_channels, *self.observation_space)
        mu, logsigma, z = self.encoder(x)

        recon_x = self.decoder(z)
        return recon_x, mu, logsigma, z

    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def _get_reconstruction_loss(self, batch, recon_x):
        return F.mse_loss(recon_x, batch, reduction='sum')
    
    def _get_regularization_loss(self, logsigma, mu):
        return -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())

    def training_step(self, batch, batch_idx):
        recon_x, mu, logsigma, _ = self.forward(batch)
        
        recon_loss = self._get_reconstruction_loss(batch, recon_x)
        reg_loss = self._get_regularization_loss(logsigma, mu)
        loss = recon_loss + reg_loss
        
        self.log("train_loss", loss)
        self.log("recon_loss", recon_loss)
        self.log("reg_loss", reg_loss)
        
        return loss


    def validation_step(self, batch, batch_idx):
        recon_x, mu, logsigma, _ = self.forward(batch)
        
        recon_loss = self._get_reconstruction_loss(batch, recon_x)
        reg_loss = self._get_regularization_loss(logsigma, mu)
        loss = recon_loss + reg_loss

        self.log("train_loss", loss)
        self.log("recon_loss", recon_loss)
        self.log("reg_loss", reg_loss)

        return {'val_loss': loss}


    def test_step(self, batch, batch_idx):
        recon_x, mu, logsigma, _ = self.forward(batch)
        
        recon_loss = self._get_reconstruction_loss(batch, recon_x)
        reg_loss = self._get_regularization_loss(logsigma, mu)
        loss = recon_loss + reg_loss

        self.log("train_loss", loss)
        self.log("recon_loss", recon_loss)
        self.log("reg_loss", reg_loss)
        return loss
    
    def get_as_transform(self):
        return VAE.VAETransform(self.encoder)
    
    class VAETransform(object):
        """Rescale the image in a sample to a given size.

        Args:
            output_size (tuple or int): Desired output size. If tuple, output is
                matched to output_size. If int, smaller of image edges is matched
                to output_size keeping aspect ratio the same.
        """
        def __init__(self, encoder) -> None:
            self.encoder = encoder

        def __call__(self, sample):
            _, _, z = self.encoder(sample)
                
            return z.view(-1).clone().detach()