from src import throughput

import numpy as np
import numpy.linalg as LA
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from astropy import constants as const
from astropy import units as u
from astropy.io import fits
from scipy.ndimage.filters import gaussian_filter
from PyAstronomy import pyasl
import matplotlib.pyplot as plt
import pandas as pd
import coronagraph as cg
from astropy.convolution import Gaussian1DKernel
import os

path_file = os.path.dirname(__file__)
load_path = os.path.join(os.path.dirname(path_file), "sim_data/Spectra/")


class Spectrum:

    def __init__(self, wavelength, flux, R, T):
        self.wavelength = wavelength
        self.flux = flux
        self.R = R
        self.Temperature = T
        self.high_pass_flux = None

    def set_flux(self, nbPhotons):
        self.flux = nbPhotons * self.flux / np.sum(self.flux)

    def degrade_resolution(self, Rnew, wave, tell=False):
        dwl = np.zeros_like(wave) + (wave[1]-wave[0])
        valid = np.where((self.wavelength >= wave[0]) & (self.wavelength <= wave[-1]))
        flr = cg.downbin_spec(self.flux, self.wavelength, wave, dlam=dwl)
        if tell:
            return Spectrum(wave, flr, Rnew, self.Temperature)
        else:
            return Spectrum(wave, np.sum(self.flux[valid])*flr/np.nansum(flr), Rnew, self.Temperature)

    def interpolate_wavelength(self, influx, wavelength_input, wavelength_output, nbPhot=None):
        """
        Interpolate a 1d array over a new axis
        """
        if nbPhot is None:
            nbPhot = np.sum(influx[np.where((wavelength_input >= wavelength_output[0]) & (wavelength_input <= wavelength_output[-1]))])
        # Interpolate

        flog = np.log10(influx)
        f = interp1d(wavelength_input, flog, bounds_error=False, fill_value=0)
        finterp_log = f(wavelength_output)
        flux_interp = 10. ** finterp_log
        return Spectrum(wavelength_output, nbPhot * (flux_interp / np.sum(flux_interp)), self.R, self.Temperature)

    def doppler_shift(self, rv):
        """
        Doppler shift a spectrum
        :rv: radial velocity (in km/s)
        """

        rv = rv * (u.km / u.s)
        rv = rv.to(u.m / u.s)
        # Compute the shifted wavelength array
        wshift = self.wavelength * (1 + (rv / const.c))
        # Compute the flux at the original wavelength sampling
        return self.interpolate_wavelength(self.flux, wshift, self.wavelength)

    def broad(self, broad):
        self.flux = pyasl.fastRotBroad(self.wavelength * 1e4, self.flux, 0.8, broad)

    def broad_r(self, broad):
        flux = pyasl.fastRotBroad(self.wavelength * 1e4, self.flux, 0.8, broad)
        return Spectrum(self.wavelength, flux, self.R, self.Temperature)

    def template_projection_tell(self, fraction_PSF, transmission=1, sigma=10):
        flux = np.copy(self.flux) * fraction_PSF
        self.high_pass_flux = (np.copy(flux)  - gaussian_filter(np.copy(flux) , sigma=sigma)) * transmission
        template = self.high_pass_flux / np.linalg.norm(self.high_pass_flux)
        alpha = np.dot(self.high_pass_flux, template)

        return alpha

    def template_projection_tell_wo_filter(self, fraction_PSF, transmission=1):
        flux = np.copy(self.flux) * fraction_PSF * transmission
        alpha = np.linalg.norm(flux)

        return alpha

    def plot_spectrum(self, show=True):
        plt.figure()
        plt.plot(self.wavelength, self.flux)
        if show:
            plt.show()


    def plot_psd(self, sigma=200, show=True, zeropadding=1, color='b', hatch="//"):
        self.high_pass_flux = (self.flux - gaussian_filter(self.flux, sigma=sigma))
        N0 = len(self.high_pass_flux)
        N = N0 * zeropadding
        ffreq = np.fft.fftfreq(N)
        sig = np.zeros(N)
        sig[:N0] = self.high_pass_flux
        fft = np.abs(np.fft.fft(sig))**2
        label= "T="+str(self.Temperature)+"K, R="+str(int(self.R))
        plt.title("Power Spectrum Density", fontsize=14)
        DSP = fft[:N // 2] * (1 / N)
        DSP_smooth = gaussian_filter(DSP, sigma=200)

        plt.plot(ffreq[:N // 2]*100000*2, DSP_smooth, label=label, color=color)
        plt.fill_between(ffreq[:N // 2]*100000*2, DSP_smooth, 0, color="none", hatch=hatch, edgecolor=color,
                           label=r'$\alpha$²: '+"T="+str(self.Temperature)+"K, R="+str(int(self.R)))
        plt.legend()
        plt.xlabel("Resolution", fontsize=14)
        plt.ylabel(r'$|S_{res}(Resolution)|²$', fontsize=14)
        plt.xlim([10, 100000])
        plt.xscale('log')
        if show:
            plt.show()




def read_fits(filepath):
    hdul = fits.open(filepath)
    model = hdul[1].data
    resolution = 100000.
    wave = np.array(model['Wavelength'])
    if len(wave.shape) == 2:
        wave = np.reshape(wave, wave.shape[1])
    flx = np.array(model['Flux'])
    if len(flx.shape) == 2:
        flx = np.reshape(flx, flx.shape[1])
    spectrum = np.array([wave * 1e3, flx])
    return spectrum, resolution


def load_spectrum(T, lg=3.5, load_path=load_path):
    lg0 = [3.5, 4.0, 4.5, 5.0, 5.5]
    if lg not in lg0:
        array = np.asarray(lg0)
        idx = (np.abs(array - lg)).argmin()
        lg = array[idx]
    print(lg)
    load_path += '/lte-g' + str(lg) + '/'
    list_spectra = os.listdir(load_path)

    if T < 500:
        T = 500
        print("Changing the input temperature to the minimal temperature : 500K.")
    if T > 2800:
        T = 2800
        print("Changing the input temperature to the maximal temperature : 2800K.")

    input_temp = []
    for name in list_spectra:
        input_temp.append(int(name[3:7]))
    lst = np.asarray(input_temp)
    idx = (np.abs(lst - T)).argmin()

    spectrum, resolution = read_fits(load_path + list_spectra[idx])
    spec = Spectrum(spectrum[0, :] / 1000, spectrum[1, :] + 0.001, resolution, T)

    return spec


def load_exorem(T, lg, M, CO=0.10, load_path=load_path):
    T0 = np.arange(400, 2000, 50)
    if T not in T0:
        array = np.asarray(T0)
        idx = (np.abs(array - T)).argmin()
        T = array[idx]

    lg0 = [3.5, 4.0, 4.5, 5.0, 5.5]
    if lg not in lg0:
        array = np.asarray(lg0)
        idx = (np.abs(array - lg)).argmin()
        lg = array[idx]

    M0 = [0.32, 1.00, 3.16, 10.00]
    if M not in M0:
        array = np.asarray(M0)
        idx = (np.abs(array - M)).argmin()
        M = array[idx]

    CO0 = np.arange(0.10, 0.75, 0.05)
    if CO not in CO0:
        array = np.asarray(CO0)
        idx = (np.abs(array - CO)).argmin()
        CO = array[idx]

    prefix = "spectra_YGP_" + str(T) + "K_logg" + f'{lg:.1f}' + "_met" + f'{M:.2f}' + "_CO" + f'{CO:.2f}' + ".dat"

    data = np.loadtxt(load_path + "exorem/" + prefix)
    flux = data[:, 1] * (data[:, 0] ** 2) / 1e4
    wave = 1e4 / data[:, 0]

    ord = np.argsort(wave)
    wave_ord = wave[ord]
    flux_ord = flux[ord]
    R = 100000  # 2 / np.abs(np.mean(np.diff(wave)))
    delta_lamb = np.mean(wave_ord) / 2 * R
    E_phot = (const.h * const.c) / wave_ord
    flux_ord *= delta_lamb / E_phot.value

    spectrum = Spectrum(wave_ord, flux_ord, R, T)

    return spectrum


def read_bz2(filepath):
    dataf = pd.pandas.read_csv(filepath,
                               usecols=[0, 1],
                               names=['wavelength', 'flux'],
                               header=None,
                               dtype={'wavelength': str, 'flux': str},
                               delim_whitespace=True,
                               compression='bz2')
    dataf['wavelength'] = dataf['wavelength'].str.replace('D', 'E')
    dataf['flux'] = dataf['flux'].str.replace('D', 'E')
    dataf = dataf.apply(pd.to_numeric)
    data = dataf.values
    star_wavel = data[:, 0] * 1e-4  # (Angstrom) -> (um)
    star_spect = 10. ** (data[:, 1] - 8.)  # (erg s-1 cm-2 Angstrom-1)

    index_sort = np.argsort(star_wavel)
    spectrum = np.array([star_wavel[index_sort], star_spect[index_sort]])
    resolution = 200000.

    spec = Spectrum(spectrum[0, :], spectrum[1, :] + 0.001, resolution, T=None)
    return spec
