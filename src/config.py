import collections
import numpy as np

'''File that stores hardwired data for use in HARMONI
    pipeline. Data stored in dictionary format
    with keywords.
    '''

GratingInfo = collections.namedtuple('GratingInfo', 'lmin, lmax, R')
ApodizerInfo = collections.namedtuple('ApodizerInfo', 'transmission, sep')

config_data_HARMONI = {
    'name': "HARMONI",
    'telescope': {"diameter": 37, "area": 980.},
    # all-glass diameter in m, including central hole, secondary and spider obscuration in m^2, K, Armazones coord.
    'data_dir': "tp_data/",

    'gratings': {"HK": GratingInfo(1.450, 2.450, 3355.),
                 "H": GratingInfo(1.435, 1.815, 7104.),
                 "H_high": GratingInfo(1.538, 1.678, 17385.),
                 "K": GratingInfo(1.951, 2.469, 7104.),
                 "K1_high": GratingInfo(2.017, 2.20, 17385.),
                 "K2_high": GratingInfo(2.199, 2.40, 17385.)},

    'apodizers': {"SP1": ApodizerInfo(0.45, 70), "SP2": ApodizerInfo(0.35, 100), "SP3": ApodizerInfo(0.53, 50), "SP4": ApodizerInfo(0.59, 30), "NO_SP": ApodizerInfo(0.84, 50)},
    'strehl': {"JQ1", "JQ2", "MED"},

    'spec': {"RON": 10.0, "dark_current": 0.0053, "pixscale": 0.004, "minDIT": 0.026, "maxDIT": 1, "saturation_e": 40000., "Q_eff": 0.90},
    # e-,e-, e-/s,px/arcsec,%,%,%, K, K, e-; ron_longexp for DIT>120s

}

config_data_ERIS = {
    'name': "ERIS",
    'telescope': {"diameter": 8, "area": 49.3},
    # all-glass diameter in m, including central hole, secondary and spider obscuration in m^2, K, Armazones coord.

    'data_dir': "tp_data/",

    'gratings': {"J_low": GratingInfo(1.09, 1.42, 5000.),
                 "H_low": GratingInfo(1.45, 1.87, 5200.),
                 "K_low": GratingInfo(1.93, 2.48, 5600.),
                 "J_short": GratingInfo(1.10, 1.27, 10000.),
                 "J_middle": GratingInfo(1.18, 1.35, 10000.),
                 "J_long": GratingInfo(1.26, 1.43, 10000.),
                 "H_short": GratingInfo(1.46, 1.67, 10400.),
                 "H_middle": GratingInfo(1.56, 1.77, 10400.),
                 "H_long": GratingInfo(1.66, 1.87, 10400.),
                 "K_short": GratingInfo(1.93, 2.22, 11200.),
                 "K_middle": GratingInfo(2.06, 2.34, 11200.),
                 "K_long": GratingInfo(2.19, 2.47, 11200.)},


    'apodizers': {"NO_SP": ApodizerInfo(1, 0)},
    'strehl': {"JQ1"},

    'spec': {"RON": 12.0, "dark_current": 0.1, "pixscale": 0.025, "minDIT": 0.026, "maxDIT": 2, "saturation_e": 40000., "Q_eff": 0.85},
    # e-, e-/s, arcsec/pxl, min, e-, e-/ADU, % ;

}

config_data_list = [config_data_HARMONI, config_data_ERIS]

