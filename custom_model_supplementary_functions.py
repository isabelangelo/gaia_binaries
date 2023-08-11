"""
defines the following functions needed for custom model:
    flux_weights: determines the weights of the primary and secondary to be used in the binary model
    training_set_density: used in iterative training to remove anomalous stars
"""
import astropy.constants as c
import astropy.units as u
import pandas as pd
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d

########## calculate relative flux weights of primary,secondary ###############

# speed of light for doppler shift calculation
speed_of_light_kms = c.c.to(u.km/u.s).value

# load data from Pecuat & Mamajek for flux/magnitude calculations
pm2013 = pd.read_csv('./data/literature_data/PecautMamajek_table.csv', 
                    delim_whitespace=True).replace('...',np.nan)

# get Vmag, V-I flux from Pecaut & Mamajek
teff_pm2013 = np.array([float(i) for i in pm2013['Teff']])
VminusI_pm2013 = np.array([float(i) for i in pm2013['V-Ic']])
V_pm2013 = np.array([float(i) for i in pm2013['Mv']])
mass_pm2013 = np.array([float(i) for i in pm2013['Msun']])
teff2Vmag = interp1d(teff_pm2013, V_pm2013)

# interpolate between columns
valid_mass = ~np.isnan(mass_pm2013)
teff2Vmag = interp1d(teff_pm2013[valid_mass], V_pm2013[valid_mass])
teff2VminusI = interp1d(teff_pm2013[valid_mass],VminusI_pm2013[valid_mass])
teff2mass = interp1d(teff_pm2013[valid_mass], mass_pm2013[valid_mass])

def flux_weights(teff1, teff2):
    # compute relative I band flux > flux ratio
    I1 = teff2Vmag(teff1) - teff2VminusI(teff1)
    I2 = teff2Vmag(teff2) - teff2VminusI(teff2)
    f1_over_f2 = 10**((I2-I1)/2.5)
    flux2_weight = 1/(1+f1_over_f2)
    flux1_weight = 1-flux2_weight
    return(flux1_weight, flux2_weight)


########## compute density of training set for a given set of model parameters ########## 

training_label_df = pd.read_csv('./data/label_dataframes/training_labels.csv')
def training_set_density(cannon_params):
    teff, logg, feh, alpha, vbroad = cannon_params

    # these are kind of randomly chosen at the momend
    # teff_window = 100
    # logg_window = 0.1
    # feh_window = 0.1
    # alpha_window = 0.1
    # vbroad_window = 5
    teff_window = 500
    logg_window = 0.25
    feh_window = 0.25
    alpha_window = 0.1
    vbroad_window = 10

    training_neighbors = training_label_df.query(
        'galah_teff < @teff+@teff_window & galah_teff > @teff-@teff_window \
        & galah_logg < @logg+@logg_window & galah_logg > @logg-@logg_window \
        & galah_feh < @feh+@feh_window & galah_feh > @feh-@feh_window \
        & galah_alpha < @alpha+@alpha_window & galah_alpha > @alpha-@alpha_window \
        & galah_vbroad < @vbroad+@vbroad_window & galah_vbroad > @vbroad-@vbroad_window')

    hypercube_volume = teff_window*logg_window*feh_window*alpha_window*vbroad_window
    density = len(training_neighbors)/hypercube_volume
    return density



