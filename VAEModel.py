from VAE_spectra import SpectralModel, SpectralData
from sklearn import preprocessing
import numpy as np
import torch
from torch.utils.data import Dataset


class StellarDataCapsule(Dataset):

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

        spectrum = self.m_data.m_spectra[idx, :]
        spectrum_ivar = self.m_data.m_spectra_ivar[idx, :]

        label = self.m_data.m_label[idx, :]
        label_ivar = self.m_data.m_label_ivar[idx, :]

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

    def normalize_new_data(self,
                           stellar_data):

        return SpectralData(self.m_wavelength,
                            stellar_data.m_spectra - self.m_spectral_mean,
                            stellar_data.m_spectra_ivar,
                            self.m_norm_model_label.transform(stellar_data.m_label),
                            stellar_data.m_label_ivar * self.m_norm_model_label.var_,
                            label_names=self.m_label_names)

    # TODO de-normalize the results


class ToTensor(object):

    def __call__(self, sample):
        spectrum, spectrum_ivar = sample['spectra'], sample['spectra_ivar']
        label, label_ivar = sample['labels'], sample['label_ivar']

        return {'spectra': torch.from_numpy(spectrum).view(1, 1, -1),
                'spectra_ivar': torch.from_numpy(spectrum_ivar).view(1, 1, -1),
                'labels': torch.from_numpy(label).view(1, -1),
                'label_ivar': torch.from_numpy(label_ivar).view(1, -1)}


class VAEModel(SpectralModel):

    def __init__(self):
        pass

    def train(self,
              train_dataset):
        pass

    def infer_labels_from_spectra(self,
                                  fluxes_in,
                                  flux_vars_in):
        pass

    def infer_spectra_from_labels(self,
                                  labels_in,
                                  label_vars_in):
        pass