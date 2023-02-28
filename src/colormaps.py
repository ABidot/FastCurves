from src.spectrum import *
from src.config import config_data_list
from src.contrast_curves import get_config_data

import os
import matplotlib.pyplot as plt
import numpy as np

path_file = os.path.dirname(__file__)
save_path = os.path.join(os.path.dirname(path_file), "output_curves/")
save_path_colormap = os.path.join(os.path.dirname(path_file), "colormaps/")

def colormap(T, step_l0=0.1, nbPixels=3330, log=False, tellurics=True, broadening=0,
             save=True,
             save_path=save_path_colormap,
             show=False, ret=False, model="BT_Settl", instru="HARMONI"):
    """
    Creating figure for IFS Trade-off between Bandwidth/Resolution
    :param T: Temperature of the companion
    :param step_l0: sampling step for the wavelength axis
    :param nbPixels: number of pixels considered to sample a spectrum
    :param log: logscale for the y-axis of the figure if True
    :param tellurics: tellurics absorption considered if True
    :param broadening: rotational broadening (km/s)
    :param save: save the figure if True
    :param save_path: saving path for the figure
    :param show: displays the figures if True
    :param ret: return the colormaps array if True
    :param model: name of the template library ("BT_Settl" or "ExoREM")
    :param instru: name of the instrument considered
    """

    R = np.logspace(2.7, 4.99, num=80)

    lambda_0 = np.arange(1., 2.8, step_l0)
    alpha = np.zeros((lambda_0.shape[0], R.shape[0]), dtype=float)
    noise = np.zeros_like(alpha, dtype=float)

    path_file = os.path.dirname(__file__)
    mol_star = read_bz2(os.path.join(os.path.dirname(path_file), "sim_data/Spectra/lte080-4.0-0.0a+0.0.BT-NextGen.7.bz2"))
    if model == "BT_Settl":
        planet = load_spectrum(T, lg=4.0)
    elif model == "ExoREM":
        planet = load_exorem(T, 4.0, 1, CO=0.60)

    Nb_Photons_tot = 8.89e6 * 980 * 60 *  1.8 * 1000
    delta_lamb = 2 / (2 * 100000)
    wave_output = np.arange(0.7, 3, delta_lamb)

    mol_star = mol_star.interpolate_wavelength(mol_star.flux, mol_star.wavelength, wave_output)
    mol_star.flux = np.zeros_like(mol_star.flux) + 1
    mol_star.set_flux(Nb_Photons_tot)

    planet_contrast = planet.interpolate_wavelength(planet.flux, planet.wavelength, wave_output)
    if broadening != 0:
        planet_contrast.broad(broadening)
    planet_contrast.set_flux(Nb_Photons_tot * 10 ** (-5))

    sky_transmission_path = os.path.join(os.path.dirname(path_file), "sim_data/Transmission/sky_transmission_airmass_1.fits")
    sky_trans = fits.getdata(sky_transmission_path)
    sky_trans = Spectrum(sky_trans[0, :], sky_trans[1, :], 100000, None)

    for j, res in enumerate(R):
        delta_lamb = 2 / (2 * res)
        wav = np.arange(0.7, 3, delta_lamb)
        planet_R = planet_contrast.degrade_resolution(res, wav)
        star_contrast = mol_star.degrade_resolution(res, wav)
        sky_R = sky_trans.degrade_resolution(res, wav, tell=True)
        sigma = np.sqrt(np.log(2)) / (np.pi * (245 / (2 * res)))
        sigma = max(1, sigma)

        for i, l0 in enumerate(lambda_0):
            umin = l0 - (nbPixels // 2) * delta_lamb
            umax = l0 + (nbPixels // 2) * delta_lamb
            valid = np.where(((wav < umax) & (wav > umin)))
            if tellurics:
                trans = sky_R.flux[valid]
            else:
                trans = 1
            noise[i, j] = np.sqrt(np.mean(star_contrast.flux[valid] * trans))
            planet_R_crop = Spectrum(wav[valid], planet_R.flux[valid], res, T)
            alpha[i, j] = planet_R_crop.template_projection_tell(1, transmission=trans, sigma=sigma)

    xx, yy = np.meshgrid(lambda_0, R)
    if log:
        plt.yscale('log')
    z = (alpha / noise).transpose() / np.nanmax((alpha / noise))

    plt.contour(xx, yy, z, linewidths=0.5,
                colors='k')
    plt.pcolormesh(xx, yy, z,
                   cmap=plt.get_cmap('rainbow'), vmin=0, vmax=1)

    x_instru, y_instru, labels = [], [], []
    config_data = get_config_data(instru)
    for band in config_data['gratings']:
        x_instru.append((config_data['gratings'][band].lmin + config_data['gratings'][band].lmax) / 2)
        y_instru.append(config_data['gratings'][band].R)

        labels.append(band)

    if tellurics:
        plt.title(f"Trade-off for T = {planet.Temperature} K with tellurics")
    if broadening != 0:
        plt.title(f"Trade-off for T = {planet.Temperature} K, $v_b = {broadening} km/s$")


    else:
        plt.title(f"Trade-off for T = {planet.Temperature} K")
    plt.xlabel("Lambda 0 (um)")
    plt.ylabel("Resolution")
    plt.ylim([500, 100000])

    cbar = plt.colorbar()
    cbar.set_label('Normalized SNR', labelpad=20, rotation=270)
    plt.scatter(x_instru, y_instru, c='black', marker='o', label=instru + ' modes')
    for i, l in enumerate(labels):
        if log:
            plt.annotate(l, (x_instru[i] , 1.2*y_instru[i]))
        else:
            plt.annotate(l, (x_instru[i] + 0.01, y_instru[i] + 100))
    plt.legend()

    if save:
        filename = f"Colormap_{planet.Temperature} K"
        if tellurics:
            filename += "_with_tellurics"
        if broadening != 0:
            filename += f"_broadening_{broadening}_kmh"
        plt.savefig(save_path + filename + ".png", format='png')

    if show:
        plt.show()
    plt.close()

    if ret:
        return xx, yy, (alpha / noise).transpose()

def colormap_no_trade(T, step_l0=0.1, log=False, tellurics=True,
             save=True,
             save_path=save_path_colormap,
             show=False, ret=False):
    """
    Creating figure to identify the optimal spectral range for molecular mapping
    :param T: Temperature of the companion
    :param step_l0: sampling step for the wavelength axis
    :param log: logscale for the y-axis of the figure if True
    :param tellurics: tellurics absorption considered if True
    :param broadening: rotational broadening (km/s)
    :param save: save the figure if True
    :param save_path: saving path for the figure
    :param show: displays the figures if True
    :param ret: return the colormaps array if True
    """

    R = np.logspace(2, 4.99, num=100)

    lambda_0 = np.arange(1, 2.7, step_l0)
    alpha = np.zeros((lambda_0.shape[0], R.shape[0]), dtype=float)
    noise = np.zeros_like(alpha, dtype=float)

    path_file = os.path.dirname(__file__)
    mol_star = read_bz2(os.path.join(os.path.dirname(path_file), "sim_data/Spectra/lte080-4.0-0.0a+0.0.BT-NextGen.7.bz2"))
    planet = load_spectrum(T, lg=4.0)

    sky_transmission_path = os.path.join(os.path.dirname(path_file),
                                         "sim_data/Transmission/sky_transmission_airmass_1.fits")
    sky_trans = fits.getdata(sky_transmission_path)
    sky_trans = Spectrum(sky_trans[0, :], sky_trans[1, :], 100000, None)

    # planet = load_exorem("C:/Users/bidotal/Downloads/spectra_YGP_1000K_logg4.0_met1.00_CO0.75.dat")
    Nb_Photons_tot = 8.89e6 * 2.3 * 1000 * 980 * 60
    delta_lamb = 2 / (2 * 100000)
    wave_output = np.arange(0.7, 3, delta_lamb)

    mol_star = mol_star.interpolate_wavelength(mol_star.flux, mol_star.wavelength, wave_output)
    mol_star.set_flux(Nb_Photons_tot)
    planet_contrast = planet.interpolate_wavelength(planet.flux, planet.wavelength, wave_output)
    planet_contrast.set_flux(Nb_Photons_tot * 10 ** (-5))

    for j, res in enumerate(R):
        print(res)
        delta_lamb = 2 / (2 * res)
        wav = np.arange(0.7, 3, delta_lamb)
        planet_R = planet_contrast.degrade_resolution(res, wav)
        star_contrast = mol_star.degrade_resolution(res, wav)
        sigma = np.sqrt(np.log(2)) / (np.pi * (245 / (2 * res)))
        sigma = max(1, sigma)

        if tellurics:
            sky_R = sky_trans.degrade_resolution(res, wav, tell=True)

        for i, l0 in enumerate(lambda_0):
            umin = l0 - 0.25
            umax = l0 + 0.25
            valid = np.where(((wav < umax) & (wav > umin)))
            if tellurics:
                trans = sky_R.flux[valid]
            else:
                trans = 1

            noise[i, j] = np.sqrt(np.mean(star_contrast.flux[valid] * trans))
            planet_R_crop = Spectrum(wav[valid], planet_R.flux[valid], res, T)
            alpha[i, j] = planet_R_crop.template_projection_tell(1, transmission=trans, sigma=sigma)

    if log:
        plt.yscale('log')

    z = (alpha / noise).transpose() / np.nanmax((alpha / noise))
    xx, yy = np.meshgrid(lambda_0, R)

    plt.contour(xx, yy, z, linewidths=0.5,
                colors='k')
    plt.pcolormesh(xx, yy, z,
                   cmap=plt.get_cmap('rainbow'))
    plt.ylim([100, 100000])

    plt.title(f"SNR gain for T ={planet.Temperature} K spectrum type")
    plt.xlabel("Lambda 0 (µm)")
    plt.ylabel("Resolution")

    cbar = plt.colorbar()
    cbar.set_label('Normalized SNR', labelpad=20, rotation=270)

    if save:
        filename = f"Colormap_{planet.Temperature} K"
        plt.savefig(save_path + filename + "no_trade.png", format='png')

    if show:
        plt.show()

    plt.close()

    if ret:
        return xx, yy, (alpha / noise).transpose()

