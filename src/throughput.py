from src.spectrum import *
from src.config import config_data_HARMONI

from astropy.io import ascii
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from astropy import constants as const
from astropy import units as u
import numpy as np
import os

path_file = os.path.dirname(__file__)
tppath = os.path.join(os.path.dirname(path_file), "sim_data/Transmission/HARMONI/Instrumental_transmission/")
vegapath = os.path.join(os.path.dirname(path_file), "sim_data/alpha_lyr_005.fits")


def telescope_throughput(waveobs, ao=True):
    """
    Compute the telescope transmission at the observed wavelengths
    Args:
        waveobs: numpy 1d array of N elements, input wavelengths of interest [micron]
        ao: Boolean, consider the transmission of the AO dichroic into the total optical path [default=True]
    Returns:
        tel_tr_interpolated: numpy 1d array of N elements, telescope transmission at the observed wavelengths
    """

    # Load the mirror transmission curve
    tel_tr = ascii.read(os.path.join(tppath, 'ELT_mirror_reflectivity.txt'))

    # Interpolate onto the input wavelength array
    f = interp1d(tel_tr['col1'], tel_tr['col2'],  bounds_error=False, fill_value=0)
    tel_tr_interpolated = f(waveobs)

    if ao is True:
        # Load the dichroic emission curve which is used to feed the AO system
        ao_tr = ascii.read(os.path.join(tppath, 'ao_dichroic.txt'))
        # Interpolate onto the input wavelength array
        f = interp1d(ao_tr['col1'], ao_tr['col2'],  bounds_error=False, fill_value=0)
        ao_tr_interpolated = 1. - f(waveobs)  # transmission
        tel_tr_interpolated = np.multiply(tel_tr_interpolated, ao_tr_interpolated)  # Combined both transmission curves
    return tel_tr_interpolated


def fprs_throughput(waveobs):
    """
    Compute the FPRS transmission at the observed wavelengths
    Args:
        waveobs: numpy 1d array of N elements, input wavelengths of interest [micron]
    Returns:
        fprs_tr_interpolated: numpy 1d array of N elements, FPRS transmission at the observed wavelengths
    """
    # Load the Focal plane relay system, not taken into account into the instrument grating transmission curve
    fprs_tr = ascii.read(os.path.join(tppath, 'FPRS.txt'))

    # Interpolate onto the input wavelength array
    f = interp1d(fprs_tr['col1'], fprs_tr['col2'],  bounds_error=False, fill_value=0)
    fprs_tr_interpolated = f(waveobs)

    return fprs_tr_interpolated


def instrument_throughput(waveobs, filter, CRYOSTAT=True):
    """
    Compute the instrument transmission at the observed wavelengths
    Args:
        waveobs: numpy 1d array of N elements, input wavelengths of interest [micron]
        filter: string, name of the filter of the observations
        CRYOSTAT: boolean, if set, consider the transmission of the HARMONI pre-IFU optics, IFU, and spectrograph [default=True]
    Returns:
        instrument_tr_interpolated: numpy 1d array of N elements, instrument transmission at the observed wavelengths
    """

    # Load the specific grating emission curve
    l_grating, emi_grating = np.loadtxt(os.path.join(tppath, filter + '_grating.txt'), unpack=True,
                                        comments="#", delimiter=",")
    emi_grating *= ((config_data_HARMONI['telescope']['diameter'] / 2.) ** 2. * np.pi / config_data_HARMONI['telescope'][
        'area'])  # scaling with effective surface

    # Interpolate onto the input wavelength array
    f = interp1d(l_grating, emi_grating, bounds_error=False, fill_value=0)
    emi_grating_interpolated = f(waveobs)

    instrument_tr_interpolated = 1. - emi_grating_interpolated

    if CRYOSTAT is True:
        # Load the lenses transmission profiles, not taken into account into the instrument grating transmission curve
        l_lens, emi_lens = np.loadtxt(os.path.join(tppath, 'HARMONI_lens_emissivity.txt'), unpack=True,
                                      comments="#", delimiter=",")
        cryo_lens_emi = 1. - (1. - emi_lens) ** 8  # 8 lenses in the cryostat, hard coded from HSIM
        cryo_lens_emi *= ((config_data_HARMONI['telescope']['diameter'] / 2.) ** 2. * np.pi / config_data_HARMONI['telescope'][
            'area'])  # scaling with effective surface
        # Interpolate over the input wavelength array
        f = interp1d(l_lens, cryo_lens_emi, bounds_error=False, fill_value=0)
        cryo_lens_emi_interpolated = f(waveobs)
        # Load the mirrors transmission profiles, not taken into account into the instrument grating transmission curve
        l_mirror, emi_mirror = np.loadtxt(os.path.join(tppath, 'HARMONI_mirror_emissivity.txt'),
                                          unpack=True, comments="#", delimiter=",")
        cryo_mirror_emi = 1. - (1. - emi_mirror) ** 19  # 19 mirrors in the cryostat, hard coded from HSIM
        cryo_mirror_emi *= ((config_data_HARMONI['telescope']['diameter'] / 2.) ** 2. * np.pi / config_data_HARMONI['telescope'][
            'area'])  # scaling with effective surface
        # Interpolate over the input wavelength array
        f = interp1d(l_mirror, cryo_mirror_emi, bounds_error=False, fill_value=0)
        cryo_mirror_emi_interpolated = f(waveobs)
        # Compute combined transmission profile
        cryo_emi_interpolated = 1. - ((1. - cryo_mirror_emi_interpolated) * (1. - cryo_lens_emi_interpolated))
        cryo_tr_interpolated = 1. - cryo_emi_interpolated
        instrument_tr_interpolated = np.multiply(instrument_tr_interpolated,
                                                 cryo_tr_interpolated)  # Combined both transmission curves

    return instrument_tr_interpolated


def zeropoint(waveobs, config_data, channel_width):
    """
    Compute the Zero point of the instrument, i.e, the incoming flux needed to get 1 e-/s onto the detector. Args:
    waveobs: numpy 1d array of N elements, input wavelengths of interest [micron] filter: string, name of the filter
    of the observations Returns: ZP: float [with magnitude = False, numpy 1d array], incoming magnitude [flux]
    leading to 1 e-/s onto the detector throughput: transmission
    """

    # Input source is Vega
    hdulist = fits.open(os.path.join(vegapath))
    vega = hdulist[1].data
    vega.WAVELENGTH = np.multiply(vega.WAVELENGTH, 1.e-4)  # Angstrom -> micron
    vega.FLUX = np.multiply(vega.FLUX,
                            1e4 * 1e4 * 1e-7)  # ergs/s/cm2/A -> ergs/s/cm2/micron -> ergs/s/m2/micron -> W/m2/micron

    # Format Vega onto a spectrum object to ease further manipulations
    vega_resolution = np.round(np.mean(vega.WAVELENGTH) / (vega.WAVELENGTH[1] - vega.WAVELENGTH[0]))
    vega_spec = Spectrum(vega.WAVELENGTH, vega.FLUX, vega_resolution, None)

    # Degrade Vega' spectrum to the observation resolution and interpolate over the observation wavelength array
    f = interp1d(vega_spec.wavelength, vega_spec.flux, bounds_error=False, fill_value=0)
    finterp_log = f(waveobs)
    vega_spec_inter = finterp_log

    # Compute photon flux

    vega_flux = vega_spec_inter * (waveobs * 1.e-6 / (const.c * const.h)) # W/m2/micron -> photons/s/m2/micron
    vega_flux = vega_flux.value

    # Zeropoint
    S = config_data["telescope"]["area"]
    lambda_mean = np.mean(waveobs)
    vega_flux *= S * channel_width  # [ph/s]

    ZP = vega_flux

    return ZP
