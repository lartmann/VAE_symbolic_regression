import torch.nn as nn
import torch.nn.functional as F
import torch
from src.loss import LatentCorrelationLoss

# VAE n-hot encoded equation elements
class VAE_ECV(nn.Module):
    def __init__(self, latent_dims, vocab_size=16):
        super(VAE_ECV, self).__init__()
        self.latent_dims = latent_dims
        self.vocab_size = vocab_size
        # size of output layer for embedding
        self.l = 8

        self.embedding_equation = nn.Sequential(
            nn.Embedding(self.vocab_size, self.vocab_size), 
            nn.Linear(self.vocab_size, self.l),
            nn.ReLU(),
            nn.Flatten(),
            
        ) 
        self.encoder_layers_mean = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * self.l, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dims),
        )
        self.encoder_layers_logvar = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * self.l, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dims),
        )

        self.decoder_layers = nn.Sequential(
            nn.Linear(latent_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 7 * self.l),
            nn.ReLU(),
        )

        self.decode_eq = nn.Sequential(
            nn.Unflatten(1, (6, self.l)), nn.Linear(self.l, self.vocab_size)
        )

        self.encode_constants = nn.Sequential(nn.Linear(1, self.l), nn.ReLU())
        self.decode_constants = nn.Linear(self.l, 1)

    def encode(self, x_e, x_c):
        x_e = self.embedding_equation(x_e)
        x_c = self.encode_constants(x_c)
        x = torch.cat((x_e, x_c), dim=1)
        # print(x.shape)
        mean = self.encoder_layers_mean(x)
        logvar = self.encoder_layers_logvar(x)
        return mean, logvar

    def decode(self, z):
        y = self.decoder_layers(z)
        y_e, y_c = torch.split(
            y,
            [6 * self.l, self.l],
            dim=1,
        )
        y_e = self.decode_eq(y_e)
        y_c = self.decode_constants(y_c)

        return y_e, y_c
    
    def reparameterize(self, mean, logvar):
        # reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(self, x_e, x_c):
        mean, logvar = self.encode(x_e, x_c)
        z = self.reparameterize(mean, logvar)
        y = self.decode(z)
        return y, z, mean, logvar

# Loss function VAE_ECV
def vae_ecv_loss(recon_equations, recon_constants, equations, constants, values, z, mean, logvar, kl_weight):
    criterion = nn.CrossEntropyLoss()
    reconstruction_loss_equations = criterion(
        recon_equations.view(-1, recon_equations.size(-1)), equations.view(-1)
    ) 
    reconstruction_loss_constants = nn.MSELoss()(recon_constants, constants)
    loss_latent_correlation = LatentCorrelationLoss()(values[:, 1, :], z)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) 
    return loss_latent_correlation+ reconstruction_loss_equations + reconstruction_loss_constants  + kl_divergence * kl_weight, (reconstruction_loss_equations, reconstruction_loss_constants, loss_latent_correlation, kl_divergence * kl_weight)

# AE n-hot encoded equation elements
class AutoencoderEquations(nn.Module):
    def __init__(self, latent_dims, vocab_size=16):
        super(AutoencoderEquations, self).__init__()
        self.latent_dims = latent_dims
        self.vocab_size = vocab_size
        # size of output layer for embedding
        self.l = 8

        self.embedding_equation = nn.Sequential(
            nn.Embedding(self.vocab_size, self.vocab_size), 
            nn.Linear(self.vocab_size, self.l),
            nn.ReLU(),
            nn.Flatten(),
            
        )
        self.encoder_layers = nn.Sequential(
            #nn.Flatten(),
            nn.Linear(7 * self.l, 64),
            nn.ReLU(),
            nn.Linear(64, self.latent_dims),
        )

        self.decoder_layers = nn.Sequential(
            nn.Linear(self.latent_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 7 * self.l),
        )

        self.decode_eq = nn.Sequential(
            nn.Linear(self.l, self.vocab_size), 
        )

        self.encode_constants = nn.Sequential(nn.Linear(1, self.l))
        self.decode_constants = nn.Sequential(nn.Linear(self.l, 1))

    def encode(self, x_e, x_c):
        x_e = self.embedding_equation(x_e)
        x_c = self.encode_constants(x_c)
        x = torch.cat((x_e, x_c), dim=1)
        # print(x.shape)
        z = self.encoder_layers(x)
        return z

    def decode(self, z):
        y = self.decoder_layers(z)
        y_e, y_c = torch.split(
            y,
            [6 * self.l, self.l],
            dim=1,
        )
        y_e = torch.reshape(y_e, (-1, 6, self.l))
        y_e = self.decode_eq(y_e)
        y_c = self.decode_constants(y_c)

        return y_e, y_c

    def forward(self, x_e, x_c):
        z = self.encode(x_e, x_c)
        y = self.decode(z)
        return y, z

# Loss function AutoencoderEquations
def ae_ec_loss(recon_equations, recon_constants, equations, constants, values, z):
    criterion = nn.CrossEntropyLoss()
    reconstruction_loss_equations = criterion(
        recon_equations.view(-1, recon_equations.size(-1)), equations.view(-1)
    )
    reconstruction_loss_constants = nn.MSELoss()(recon_constants, constants)
    loss = LatentCorrelationLoss()(values[:, 1, :], z)
    return loss + reconstruction_loss_equations +  reconstruction_loss_constants


# AE one-hot encoded funtion terms
class AutoencoderEquationsClassify(nn.Module):
    def __init__(self, latent_dims, num_symbols, max_length, num_classes):
        super(AutoencoderEquationsClassify, self).__init__()
        self.latent_dims = latent_dims
        self.num_symbols= num_symbols
        self.max_length = max_length
        self.num_classes = num_classes
        # size of output layer for embedding
        self.l = 8


        self.embedding_equation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.max_length * self.num_symbols, self.max_length * self.num_symbols, ),
        )
        self.encoder_layers = nn.Sequential(
            # (max_length * num_symbols + l)
            nn.Linear((self.max_length + 1)*self.l, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dims),
        )

        self.decoder_layers = nn.Sequential(
            nn.Linear(latent_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.max_length * self.num_symbols + self.l),
        )

        self.decode_eq = nn.Sequential(
            # output is the probability of each class
            nn.Linear(self.max_length * self.num_symbols, 64),
            #nn.ReLU(),
            nn.Linear(64, self.num_classes)
        )

        self.encode_constants = nn.Linear(1, self.l)
        self.decode_constants = nn.Linear(self.l, 1)

    def encode(self, x_e, x_c):
        x_e = F.one_hot(x_e, num_classes=self.num_symbols)
        x_e = x_e.to(torch.float32)
        x_e = self.embedding_equation(x_e)
        x_c = self.encode_constants(x_c)
        # combines the equations and constants (ratio of constants and equation could be adjusted)
        x = torch.cat((x_e, x_c), dim=1)
        z = self.encoder_layers(x)
        return z

    def decode(self, z):
        y = self.decoder_layers(z)
        y_e, y_c = torch.split(
            y,
            [self.max_length * self.num_symbols, self.l],
            dim=1,
        )
        # map to softmax layer
        y_e = self.decode_eq(y_e)
        y_c = self.decode_constants(y_c)

        return y_e, y_c

    def forward(self, x_e, x_c):
        z = self.encode(x_e, x_c)
        y = self.decode(z)
        return y, z

# Loss function AutoencoderEquationsClassify
def ae_ecv_loss_classify(recon_classes, recon_constants, values, constants, cls, z, weight=1.0):
    criterion = nn.CrossEntropyLoss()
    reconstruction_loss_equations = criterion(
        recon_classes, torch.tensor(cls)
    )
    reconstruction_loss_constants = nn.MSELoss()(recon_constants, constants)
    loss = LatentCorrelationLoss()(values[:, 1, :], z)
    return reconstruction_loss_equations +  reconstruction_loss_constants + weight* loss, (reconstruction_loss_equations, reconstruction_loss_constants, loss)


# VAE one-hot encoded funtion terms
class VAE_classify(nn.Module):
    def __init__(self, latent_dims, num_classes,num_symbols, max_length, vocab_size=16):
        super(VAE_classify, self).__init__()
        self.num_symbols= num_symbols
        self.max_length = max_length
        self.latent_dims = latent_dims
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        # size of output layer for embedding
        self.l = 8

        self.embedding_equation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.max_length * self.num_symbols, self.max_length * self.num_symbols, ),
            #nn.ReLU(),
        )
        self.encoder_layers_mean = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * self.l, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dims),
        )
        self.encoder_layers_logvar = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * self.l, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dims),
        )

        self.decoder_layers = nn.Sequential(
            nn.Linear(latent_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.max_length * self.num_symbols + self.l),
        )

        self.decode_eq = nn.Sequential(
            nn.Linear(self.max_length * self.num_symbols, 64),
            nn.Linear(64, self.num_classes)
        )

        self.encode_constants = nn.Linear(1, self.l)
        self.decode_constants = nn.Linear(self.l, 1)

    def encode(self, x_e, x_c):
        x_e = F.one_hot(x_e, num_classes=self.num_symbols)
        x_e = x_e.to(torch.float32)
        x_e = self.embedding_equation(x_e)
        x_c = self.encode_constants(x_c)
        x = torch.cat((x_e, x_c), dim=1)
        # print(x.shape)
        mean = self.encoder_layers_mean(x)
        logvar = self.encoder_layers_logvar(x)
        return mean, logvar

    def decode(self, z):
        y = self.decoder_layers(z)
        y_e, y_c = torch.split(
            y,
            [6 * self.l, self.l],
            dim=1,
        )
        y_e = self.decode_eq(y_e)
        y_c = self.decode_constants(y_c)

        return y_e, y_c
    
    def reparameterize(self, mean, logvar):
        # reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(self, x_e, x_c):
        mean, logvar = self.encode(x_e, x_c)
        z = self.reparameterize(mean, logvar)
        y = self.decode(z)
        return y, z, mean, logvar

# Loss function VAE_classify
def vae_classify_loss(recon_classes, recon_constants, cls, constants, values, z, mean, logvar, kl_weight, weight):
    criterion = nn.CrossEntropyLoss()
    reconstruction_loss_equations = criterion(
        recon_classes, torch.tensor(cls)
    )
    reconstruction_loss_constants = nn.MSELoss()(recon_constants, constants)
    loss_latent_correlation = LatentCorrelationLoss()(values[:, 1, :], z)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) 
    return weight * loss_latent_correlation+ reconstruction_loss_equations + reconstruction_loss_constants  + kl_divergence * kl_weight, (reconstruction_loss_equations, reconstruction_loss_constants, loss_latent_correlation, kl_divergence)