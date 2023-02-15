from src.contrast_curves import *
from src.colormaps import *

path_file = os.path.dirname(__file__)

def run_FastCurves():
    star_spectrum = read_bz2(os.path.join(os.path.dirname(path_file), "sim_data/Spectra/lte080-4.0-0.0a+0.0.BT-NextGen.7.bz2"))
    planet_spectrum = load_spectrum(1000, lg=4.0)
    FastCurves(120, 6, planet_spectrum, star_spectrum, broadening=0, apodizer="SP1", strehl="MED",
               tellurics=True,
               verbose=True, save=True, form="png", instru="HARMONI")


if __name__ == '__main__':
    run_FastCurves()


