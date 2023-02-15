from src.contrast_curves import *
from src.colormaps import *

path_file = os.path.dirname(__file__)

def run_FastCurves():
    star_spectrum = read_bz2(os.path.join(os.path.dirname(path_file), "sim_data/Spectra/lte080-4.0-0.0a+0.0.BT-NextGen.7.bz2"))
    planet_spectrum = load_spectrum(1000, lg=4.0)
    FastCurves(120, 6, planet_spectrum, star_spectrum, broadening=0, apodizer="SP1", strehl="MED",
               tellurics=True,
               verbose=True, save=True, form="png", instru="HARMONI")

def run_Colormaps():
    colormap(1000, step_l0=0.01, nbPixels=3330, log=True, tellurics=True, broadening=0,
             save=True,
             save_path=save_path_colormap,
             show=True, ret=False, model="BT_Settl", instru="HARMONI")

if __name__ == '__main__':
    run_Colormaps()
    run_FastCurves()


