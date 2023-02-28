from src import throughput
from src.config import config_data_list, config_data_HARMONI
from src.spectrum import *

import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.cm import get_cmap


path_file = os.path.dirname(__file__)
save_path = os.path.join(os.path.dirname(path_file), "output_curves/")


def get_config_data(instrument_name):
    """
    Get the parameters of an instrument
    :param instrument_name: name of the instrument considered
    :return: config parameters of the instrument
    """
    for dict in config_data_list:
        if dict['name'] == instrument_name:
            return(dict)
    raise NameError('Undefined Instrument Name')


def contrast_curve_DIT(time, PSF_profile, alpha0, star_spectrum, config_data, apodizer, corono = None):
    """
    Compute the contrast curves
    :param time: exposure time (minutes)
    :param PSF_profile: Profile of the PSF
    :param alpha0: total amount of useful photons for molecular mapping for a 0 contrast
    :param star_spectrum: spectrum of the star
    :param config_data: config file for the instrument considered
    :param apodizer: name of the apodizer considered
    :param corono: name of the considered coronograph
    :return: contrast curves
    """
    print("MEAN PHOTON: ", np.mean(star_spectrum.flux))
    relative_mag = np.zeros_like(PSF_profile[0, :])
    relative_mag_photon_only = np.zeros_like(PSF_profile[0, :])
    noise = np.zeros_like(PSF_profile[0, :])

    instru = config_data["name"]
    quantum_efficiency = config_data["spec"]["Q_eff"]
    RON = config_data["spec"]["RON"]
    dark_current = config_data["spec"]["dark_current"]

    if instru == "HARMONI":
        sep_min = config_data_HARMONI["apodizers"][str(apodizer)].sep
        n = np.argmin(np.abs(PSF_profile[0] - sep_min))
        print(n, PSF_profile[0,n])
        max_flux_e = PSF_profile[1, n] * np.copy(star_spectrum.flux) * quantum_efficiency
    elif instru == "ERIS":
        max_flux_e = PSF_profile[1, 0] * np.copy(star_spectrum.flux) * quantum_efficiency

    saturation_e = config_data["spec"]["saturation_e"]
    min_DIT = config_data["spec"]["minDIT"]
    max_DIT = config_data["spec"]["maxDIT"]
    DIT = saturation_e/np.max(max_flux_e)
    print("saturating DIT:", DIT*60, " minutes")

    if DIT > max_DIT :
        DIT = max_DIT
    elif DIT < min_DIT :
        DIT = min_DIT
        print("Saturated detector even with the shortest integration time")

    if DIT > 4*min_DIT:
        #Up The Ramp reading mode
        N_i = np.round(DIT/(4*min_DIT))
        RON_eff = RON/np.sqrt(N_i)

    else:
        RON_eff = RON

    if instru == 'ERIS' and RON_eff < 7:
        RON_eff = 7

    print("DIT (sec)", DIT*60, "RON", RON_eff)
    alpha0 *= DIT * quantum_efficiency
    for i in range(PSF_profile.shape[1]):
        if corono is None:
            star = PSF_profile[1, i] * np.copy(star_spectrum.flux) * DIT * quantum_efficiency
        else:
            star = PSF_profile[1, i] * np.copy(star_spectrum.flux) * DIT * quantum_efficiency * corono[1, i]

        if PSF_profile[0, i] < 70 and instru == "HARMONI" :
            sigma_halo_2 = np.mean(star) * 1e-4
            alpha0_FPM = alpha0 * 1e-4
        else :
            sigma_halo_2 = np.mean(star)
            alpha0_FPM = alpha0

        sigma_dark_current_2 = dark_current * DIT * 60
        sigma_ron_2 = RON_eff ** 2
        C = 5 * np.sqrt(9 * (sigma_halo_2 + sigma_ron_2 + sigma_dark_current_2)) / (alpha0_FPM * np.sqrt(time/DIT)) # 9 est le nombre de pixels sur lequel on integre le companion
        C_photon_only = 5 * np.sqrt(9 * (sigma_halo_2)) / (alpha0_FPM * np.sqrt(time / DIT))
        relative_mag[i] = -2.5 * np.log10(C)
        relative_mag_photon_only[i] = -2.5 * np.log10(C_photon_only)

    return relative_mag, relative_mag_photon_only

def transmission_HARMONI(uwvs, band, R, apodizer):
    # Lecture des profils de transmission
    file_path = os.path.dirname(__file__)
    sky_transmission_path = os.path.join(os.path.dirname(file_path), "sim_data/Transmission/sky_transmission_airmass_1.fits")
    sky_trans = fits.getdata(sky_transmission_path)
    trans_tell_band = Spectrum(sky_trans[0, :], sky_trans[1, :], 100000, None)
    trans_tell_band = trans_tell_band.degrade_resolution(R, uwvs, tell=True)

    trans_telescope = throughput.telescope_throughput(uwvs)
    trans_instrumental = throughput.instrument_throughput(uwvs, str(band))
    trans_fprs = throughput.fprs_throughput(uwvs)
    apo_trans = config_data_HARMONI["apodizers"][str(apodizer)].transmission
    trans = trans_telescope * trans_instrumental * trans_fprs * apo_trans

    return trans_tell_band, trans


def FastCurves(time, mag_star, planet_spectrum, star_spectrum, broadening=0, apodizer="SP1", strehl="MED", tellurics=True,
                 verbose=True, save=True, form="png", save_path=save_path, instru="HARMONI", cmap_name = "Spectral", corono = False):
    """
    Main function for computing and saving the Contrast Curves
    :param time: exposure time (minutes)
    :param mag_star: magnitude of the star considered
    :param planet_spectrum: spectrum of the planet
    :param star_spectrum: spectrum of the star
    :param broadening: rotational braodening (km/s)
    :param apodizer: name of the apodizer considered
    :param strehl: name of the expected strehl condition
    :param tellurics: True or False
    :param verbose: It displays the figures if True
    :param save: Save the figures if True
    :param form: format of the figure
    :param save_path: path for saving the figure
    :param instru: name of the considered instrument
    :param cmap_name: name of the colormap
    :param corono: Considering the case of a coronograph if True
    :return: Contrast curves with photon noise only and with all sources of noise profile_interp[0, :], mag, mag_photon_only
    """
    file_path = os.path.dirname(__file__)
    PSF_path = os.path.join(os.path.dirname(file_path), "sim_data/PSF/PSF_"+ instru +'/')
    Transmission_path = os.path.join(os.path.dirname(file_path), "sim_data/Transmission/" + instru + '/')
    config_data = get_config_data(instru)

    mag = [] # input fonction
    mag_photon_only = []
    name = []

    ## Inputs internes
    R = 100000
    lambda_min, lambda_max = 1, 2.5

    ## Calcul zeropoint
    delta_lamb = 1.7 / (2 * R)
    wave = np.arange(1., 2.5, delta_lamb)
    ZP = throughput.zeropoint(wave, config_data, delta_lamb) # photons/s pour chaque canal spectral de largeur delta_lamb
    NbPhotons_ADU_0 = np.sum(ZP) * 60 * 10 ** (-0.4 * mag_star) #photons/min intégré sur la bandwidth rapporté à une magnitude mag_star

    ## Ouverture des spectres planétaires et stellaires
    planet_spectrum = planet_spectrum.interpolate_wavelength(planet_spectrum.flux, planet_spectrum.wavelength, wave) # lamba en micron, flux en photons/m2/s/nm et sera renormalisé
    if broadening != 0:
        planet_spectrum = planet_spectrum.broad_r(broadening)
    planet_spectrum.set_flux(NbPhotons_ADU_0) #spectre normalisé au nombre total de photons sur bandwidth pour un contraste de 0

    star_spectrum = star_spectrum.interpolate_wavelength(star_spectrum.flux, star_spectrum.wavelength, wave) # lamba en micron, flux en photons/m2/s/nm et sera renormalisé
    star_spectrum.set_flux(NbPhotons_ADU_0) #spectre normalisé au nombre total de photons sur bandwidth pour mag de 0

    for band in config_data['gratings']:
        name.append(band)
        lmin = config_data['gratings'][band].lmin
        lmax = config_data['gratings'][band].lmax
        R = config_data['gratings'][band].R
        delta_lambda = ((lmin+lmax)/2) / (2 * R)
        uwvs = np.arange(lmin, lmax, delta_lambda)
        NbPixels = len(uwvs)
        print("Nombre de pixels spectraux:", NbPixels)
        sigma = np.sqrt(np.log(2)) / (np.pi * (600 / (2 * R))) # paramètre attendu pour le filtre passe haut spectral pour une resolution de coupure de 600

        if instru == 'HARMONI':
            trans_tell_band, trans = transmission_HARMONI(uwvs, band, R, apodizer)

        else :
            wave, trans = fits.getdata(Transmission_path + "Instrumental_transmission/transmission_" + band + ".fits")
            f = interp1d(wave, trans, bounds_error=False, fill_value=0)
            trans = f(uwvs) #transmission par canal spectral
            sky_transmission_path = os.path.join(Transmission_path, "sky_transmission_airmass_1.fits")
            sky_trans = fits.getdata(sky_transmission_path)
            trans_tell_band = Spectrum(sky_trans[0, :], sky_trans[1, :], 100000, None)
            trans_tell_band = trans_tell_band.degrade_resolution(R, uwvs, tell=True)

        # Lecture des spectres stellaires et planetaires
        planet_spectrum_inter = planet_spectrum.degrade_resolution(R, uwvs) #degradation de la resolution et restriction de la bandwidth par rapport aux modes observationnels
        star_spectrum_inter = star_spectrum.degrade_resolution(R, uwvs)
        star_spectrum_inter.flux *= trans * trans_tell_band.flux


        # Lecture des profils de PSF

        profile = fits.getdata(
            PSF_path + "PSF_" + band + "_" + strehl + "_" + apodizer + ".fits")
        fraction_PSF = fits.getheader(PSF_path + "PSF_" + band + "_" + strehl + "_" + apodizer + ".fits")['FC'] #fraction du flux dans une boîte de 3*3 pixels

        pixscale = config_data["spec"]["pixscale"] * 1000
        separation = np.arange(pixscale, 400, pixscale/4)
        f = interp1d(profile[0], profile[1], bounds_error=False, fill_value=0)
        profile_interp = f(separation) * pixscale
        profile_interp = np.array([separation, profile_interp])

        # Calcul du signal utile
        if tellurics:
            alpha0 = planet_spectrum_inter.template_projection_tell(fraction_PSF,
                                                                    transmission=trans_tell_band.flux * trans,
                                                                    sigma=sigma)
        else:
            alpha0 = planet_spectrum_inter.template_projection_tell(fraction_PSF, transmission=trans, sigma=sigma)

        print("Mode:", band, "SNR0:", alpha0/np.sqrt(np.mean(star_spectrum_inter.flux)), "Nombre de photons stellaires pour mag demandée:", np.sum(star_spectrum_inter.flux),
              "Nombre moyen de photons par canal spectral: ", np.mean(star_spectrum_inter.flux),
              "paramètre de filtrage (kernel):", sigma, "fraction du compagnon dans le coeur:", fraction_PSF,
              "Photons utiles (MM) du compagnon pour delta_mag de 0:", alpha0,
              "Nombre de photons dans le coeur de la PSF", fraction_PSF *
              np.sum(trans * planet_spectrum_inter.flux))

        # Calcul des courbes de contraste
        if corono:
            coro = fits.getdata(PSF_path + "PSF_" + band + "_" + strehl + "_" + apodizer + "_corono.fits")
            f = interp1d(coro[0], coro[1], bounds_error=False, fill_value=0)
            coro_interp = f(separation)
            coro_interp = np.array([separation, coro_interp])

            delta_mag, delta_mag_photon_only = contrast_curve_DIT(time, profile_interp, alpha0, star_spectrum_inter,
                                                                  config_data, apodizer, corono = coro_interp)
            mag.append(delta_mag)
            mag_photon_only.append(delta_mag_photon_only)

        else:
            delta_mag, delta_mag_photon_only = contrast_curve_DIT(time, profile_interp, alpha0, star_spectrum_inter,
                                                                  config_data, apodizer, corono=None)
            mag.append(delta_mag)
            mag_photon_only.append(delta_mag_photon_only)


    sep_min = config_data["apodizers"][str(apodizer)].sep
    if verbose or save:
        plt.figure()
        if corono is True:
            plt.title(instru + " Contrast Curve, time exposure = " + str(time) + ", star mag = " + str(
            mag_star) + ", strehl = " + str(strehl) + ", apodizer = " + str(apodizer) + " with corono")
        else:
            #plt.title(instru + " Contrast Curve, time exposure = " + str(time) + ", star mag = " + str(mag_star) + ", strehl = " + str(strehl) + ", apodizer = " + str(apodizer))
            plt.title(f'T = {planet_spectrum.Temperature} K', fontsize = 16)
        plt.xlabel("Separation (mas)", fontsize=14)
        plt.ylabel(r'$\Delta$mag', fontsize=14)
        ax = plt.gca()
        ax.invert_yaxis()
        ax.axvspan(0, sep_min, color='k', alpha=0.5, lw=0)
        if cmap_name is not None:
            cmap = get_cmap(cmap_name, len(mag))
            for i in range(len(mag)):
                ax.plot(profile_interp[0, :], mag[i], label="Band " + name[i], color=cmap(i))
                ax.plot(profile_interp[0, :], mag_photon_only[i], linestyle='dashed', color=cmap(i))
        else:
            cmap = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]
            for i in range(len(mag)):
                ax.plot(profile_interp[0, :], mag[i], label="Band " + name[i], color=cmap[i])
                ax.plot(profile_interp[0, :], mag_photon_only[i], linestyle='dashed', color=cmap[i])

        ax_legend = ax.twinx()
        ax_legend.plot([], [], '-', c='k', label='Full noise')
        ax_legend.plot([], [], '--', c='k', label='Photon noise only')

        ax.legend(loc='upper right')
        ax_legend.legend(loc='lower left')
        ax_legend.tick_params(axis='y', colors='w')

        if instru == "HARMONI":
            ax.set_ylim([18, 13])
        else:
            ax.set_ylim([17, 8])

        ax.set_xlim(0)
        plt.tight_layout()

        if save:
            if corono is True:
                plt.savefig(save_path + instru + f"Contrast Curve {planet_spectrum.Temperature} K_{time} min_{mag_star}_{strehl}_{apodizer}_with_corono." + form, format=form, bbox_inches='tight')
            else:
                plt.savefig(save_path + instru + f"Contrast Curve {planet_spectrum.Temperature} K_{time} min_{mag_star}_{strehl}_{apodizer}." + form, format=form, bbox_inches='tight')
        if verbose:
            plt.show()

    return profile_interp[0, :], mag, mag_photon_only