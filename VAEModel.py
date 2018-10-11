from VAE_spectra import SpectralModel, SpectralData

from sklearn import preprocessing
import numpy as np
from copy import deepcopy

import torch
from torch.utils.data import Dataset
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader


class StellarDataCapsule(Dataset):
    """
    Class needed to use the stellar data in Pytorch
    """

    def __init__(self,
                 stelar_data,
                 normalize=True,
                 transform=None):

        self.m_data = stelar_data
        self.m_wavelength = stelar_data.m_wavelength
        self.m_label_names = stelar_data.m_label_names
        self.m_normalize = normalize
        self.m_transform = transform

        self.m_norm_model_label = preprocessing.StandardScaler().fit(stelar_data.m_label)
        self.m_spectral_mean = np.mean(stelar_data.m_spectra)

    def __len__(self):
        return self.m_data.m_spectra.shape[0]

    def __getitem__(self, idx):

        spectrum = deepcopy(self.m_data.m_spectra[idx, :])
        spectrum_ivar = deepcopy(self.m_data.m_spectra_ivar[idx, :])

        label = deepcopy(self.m_data.m_label[idx, :])
        label_ivar = deepcopy(self.m_data.m_label_ivar[idx, :])

        if self.m_normalize:
            spectrum -= self.m_spectral_mean
            label = self.m_norm_model_label.transform(label.reshape(1, -1)).reshape(-1)
            label_ivar = label_ivar * self.m_norm_model_label.var_

        sample = {'spectra': spectrum,
                  'spectra_ivar': spectrum_ivar,
                  'labels': label,
                  'label_ivar': label_ivar}

        if self.m_transform:
            sample = self.m_transform(sample)

        return sample

    def normalize_new_spectra(self,
                              spectra):
        return spectra - self.m_spectral_mean

    def de_normalize_new_spectra(self,
                                 norm_spectra):
        return norm_spectra + self.m_spectral_mean

    def normalize_new_label(self,
                            label,
                            label_ivar):
        norm_label = self.m_norm_model_label.transform(label)
        norm_ivar = label_ivar * self.m_norm_model_label.var_

        return norm_label, norm_ivar

    def de_normalize_new_label(self,
                               norm_label,
                               norm_label_ivar):
        label = self.m_norm_model_label.inverse_transform(norm_label)
        ivar = norm_label_ivar / self.m_norm_model_label.var_

        return label, ivar


class ToTensor(object):
    """
    Class needed to use the stellar data in Pytorch
    """

    def __call__(self, sample):
        spectrum, spectrum_ivar = sample['spectra'], sample['spectra_ivar']
        label, label_ivar = sample['labels'], sample['label_ivar']

        return {'spectra': torch.from_numpy(spectrum).view(-1),
                'spectra_ivar': torch.from_numpy(spectrum_ivar).view(-1),
                'labels': torch.from_numpy(label).view(-1),
                'label_ivar': torch.from_numpy(label_ivar).view(-1)}


class SpectralVAENetwork(nn.Module):
    """
    VAE architecture
    """

    def __init__(self):
        super(SpectralVAENetwork, self).__init__()

        self.encoder_shared = nn.Sequential(nn.Linear(8575, 4000).double(),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(4000, 1000).double(),
                                            nn.ReLU(inplace=True))

        self.encoder_mu = nn.Sequential(nn.Linear(1000, 500).double(),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(500, 7).double())

        self.encoder_logvar = nn.Sequential(nn.Linear(1000, 500).double(),
                                            nn.LeakyReLU(inplace=True),
                                            nn.Linear(500, 7).double())

        self.decoder = nn.Sequential(nn.Linear(7, 500).double(),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Linear(500, 1000).double(),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Linear(1000, 4000).double(),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Linear(4000, 8575).double())

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.encoder_shared.apply(init_weights)
        self.encoder_mu.apply(init_weights)
        self.encoder_logvar.apply(init_weights)
        self.decoder.apply(init_weights)

    def encode(self, x):
        ec_shared_out = self.encoder_shared(x)

        ec_mu = self.encoder_mu(ec_shared_out)
        ec_log_var = self.encoder_logvar(ec_shared_out)

        return ec_mu, ec_log_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class KLLoss(nn.Module):
    """
    KL divergence loss needed for learning the latent labels
    """

    def forward(self,
                mu_pred,
                mu,
                log_var_pred,
                inv_var):
        comp_1 = (mu - mu_pred) ** 2 * inv_var
        comp_2 = log_var_pred.exp() * inv_var

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        zeros = torch.zeros(inv_var.size(), device=device, dtype=torch.float64)
        comp_3 = - torch.where(inv_var == 0, zeros, inv_var.log())

        KLD = -0.5 * torch.mean(1 - comp_1 - comp_2 + log_var_pred - comp_3)
        return KLD


class Chi2Loss(nn.Module):
    """
    Chi^2 loss needed to learn the spectral reconstruction
    """

    def forward(self,
                spectra_pred,
                spectra_gt,
                spectra_ivar):
        Recon_loss = torch.mean((spectra_gt - spectra_pred) ** 2 * spectra_ivar)

        return Recon_loss


class SpectralVAEModel(SpectralModel):

    def __init__(self,
                 kl_weight,
                 training_epochs=1000):

        self.m_kl_weight = kl_weight

        self.m_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.m_network = SpectralVAENetwork().to(self.m_device)
        self.m_optimizer = optim.Adam(self.m_network.parameters(),
                                      lr=0.00001,
                                      amsgrad=True)

        self.m_training_epochs = training_epochs
        self.m_data_loader = None
        self.m_training_data = None

    def train(self,
              train_dataset):

        # convert training dataset into pytroch ready dataset
        print("Preparing the data...")
        self.m_training_data = StellarDataCapsule(train_dataset,
                                                  transform=ToTensor())

        self.m_data_loader = DataLoader(self.m_training_data,
                                        batch_size=50,
                                        shuffle=True,
                                        num_workers=4)

        # train the network
        kl_criterion = KLLoss()
        recon_criterion = Chi2Loss()

        print("Training the network...")
        torch.set_grad_enabled(True)

        for epoch in range(self.m_training_epochs):

            running_loss_recon = 0.0
            running_loss_label = 0.0
            iterations = 0

            for i_batch, sample_batched in enumerate(self.m_data_loader):
                self.m_optimizer.zero_grad()

                tmp_spectra = sample_batched['spectra'].to(self.m_device)
                tmp_label = sample_batched['labels'].to(self.m_device)

                pred_spectra, label_mu_pred, label_log_var = self.m_network(tmp_spectra)

                kl_loss = kl_criterion(label_mu_pred,
                                       tmp_label,
                                       label_log_var,
                                       sample_batched['label_ivar'].to(self.m_device)) * self.m_kl_weight

                recon_loss = recon_criterion(pred_spectra,
                                             tmp_spectra,
                                             sample_batched['spectra_ivar'].to(self.m_device))

                loss = recon_loss + kl_loss

                loss.backward()
                self.m_optimizer.step()

                running_loss_label += kl_loss.item()
                running_loss_recon += recon_loss.item()
                iterations += 1

            # TODO run evaluations on an evaluation data set
            print("[" + str(epoch) + "] loss label   " + str(running_loss_label / iterations) +
                  "; loss recon   " + str(running_loss_recon / iterations))

        print("Finished the training")
        torch.set_grad_enabled(False) # we are now in prediction mode and do not keep gradients

    # TODO implement loop for large test data
    def infer_labels_from_spectra(self,
                                  fluxes_in,
                                  flux_vars_in):

        fluxes_norm = self.m_training_data.normalize_new_spectra(fluxes_in)
        fluxes_norm_cuda = torch.from_numpy(deepcopy(fluxes_norm)).to(self.m_device)

        pred_label_norm, log_ivar = self.m_network.encode(fluxes_norm_cuda)
        label_ivar = np.exp(log_ivar.cpu().detach().numpy())

        # for the moment we do not use the label ivar
        return self.m_training_data.de_normalize_new_label(pred_label_norm.cpu().detach().numpy(),
                                                           label_ivar)[0]

    def infer_spectra_from_labels(self,
                                  labels_in,
                                  label_vars_in):

        label_norm, _ = self.m_training_data.normalize_new_label(labels_in, label_vars_in)
        label_norm_cuda = torch.from_numpy(deepcopy(label_norm)).to(self.m_device)

        pred_spectra_norm = self.m_network.decode(label_norm_cuda)

        return self.m_training_data.de_normalize_new_spectra(pred_spectra_norm.cpu().detach().numpy())
