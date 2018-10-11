import numpy as np
from copy import deepcopy

from TheCannon import model, dataset
from VAE_spectra import SpectralModel


class TheCannonWrapper(SpectralModel):

    def __init__(self,
                 order):

        self.m_oder = order
        self.m_train_dataset_newtype = None
        self.m_train_dataset = None

        self.m_model = model.CannonModel(order,
                                         useErrors=False)

    def train(self,
              train_dataset):

        lis = range(train_dataset.m_label.shape[0])
        ids = ["{:02d}".format(x) for x in lis]

        self.m_train_dataset_newtype = deepcopy(train_dataset)
        self.m_train_dataset = dataset.Dataset(train_dataset.m_wavelength,
                                               ids,
                                               train_dataset.m_spectra,
                                               train_dataset.m_spectra_ivar,
                                               train_dataset.m_label,
                                               ids,
                                               train_dataset.m_spectra,
                                               train_dataset.m_spectra_ivar)
        self.m_train_dataset.set_label_names(np.array(range(train_dataset.m_label.shape[1]),
                                                      dtype=np.str))

        self.m_model.fit(self.m_train_dataset)

    def infer_labels_from_spectra(self,
                                  fluxes_in,
                                  flux_vars_in):

        lis = range(fluxes_in.shape[0])
        ids = ["{:02d}".format(x) for x in lis]
        ds_test = dataset.Dataset(self.m_train_dataset_newtype.m_wavelength,
                                  ids,
                                  self.m_train_dataset_newtype.m_spectra,
                                  self.m_train_dataset_newtype.m_spectra_ivar,
                                  self.m_train_dataset_newtype.m_label,
                                  ids,
                                  fluxes_in,
                                  1.0 / flux_vars_in)
        ds_test.set_label_names(['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7'])

        _, _ = self.m_model.infer_labels(ds_test)

        return ds_test.test_label_vals

    def infer_spectra_from_labels(self,
                                  labels_in,
                                  label_vars_in):

        self.m_train_dataset.test_label_vals = labels_in
        self.m_model.infer_spectra(self.m_train_dataset)
        return self.m_model.model_spectra