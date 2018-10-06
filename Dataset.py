import pickle
import os.path


class SpectralData(object):

    def __init__(self,
                 wavelength,
                 spectra,
                 spectra_ivar,
                 label,
                 label_ivar,
                 label_names=None):

        self.m_wavelength = wavelength
        self.m_spectra = spectra
        self.m_spectra_ivar = spectra_ivar
        self.m_label = label
        self.m_label_ivar = label_ivar
        self.m_label_names = label_names

    # for subset generation with numpy arrays
    def __getitem__(self, slice_in):
        return SpectralData(self.m_wavelength,
                            self.m_spectra[slice_in],
                            self.m_spectra_ivar[slice_in],
                            self.m_label[slice_in],
                            self.m_label_ivar[slice_in],
                            label_names=self.m_label_names)

    @classmethod
    def create_from_files(cls,
                          label_location,
                          spectra_location):

        if not os.path.isfile(label_location):
            raise ValueError("Label file not found")

        if not os.path.isfile(spectra_location):
            raise ValueError("Spectra file not found")

        # load the data
        temp_file = open(label_location, 'rb')
        tmp_xx = pickle.load(temp_file, encoding='latin1')
        label_input = tmp_xx[0]
        label_ivar = tmp_xx[1]
        temp_file.close()

        temp_file = open(spectra_location, 'rb')
        spectra = pickle.load(temp_file, encoding='latin1')
        temp_file.close()

        wavelength = spectra[:, 0, 0]
        fluxes = spectra[:, :, 1].T
        fluxes_ivars = 1.0 / (spectra[:, :, 2] ** 2).T

        return cls(wavelength,
                   fluxes,
                   fluxes_ivars,
                   label_input,
                   label_ivar)
