from Evaluation import SpectralModel


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