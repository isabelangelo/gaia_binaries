"""
defines the following functions needed for custom model:
    flux_weights: determines the weights of the primary and secondary to be used in the binary model
    training_density: used in iterative training to remove anomalous stars
"""
import astropy.constants as c
import astropy.units as u
import pandas as pd
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy import stats
import thecannon as tc
from astropy.table import Table

# =====================================================================================
# load cannon models to use
recent_model_version = tc.CannonModel.read('./data/cannon_models/gaia_rvs_model.model')
model_fileroot = 'gaia_rvs_model'

training_labels = ['galah_teff', 'galah_logg','galah_feh', 'galah_alpha', 'galah_vbroad']
training_set_table = Table.read('./data/label_dataframes/training_labels.csv', format='csv')
training_set = training_set_table[training_labels]

training_flux_df = pd.read_csv('./data/gaia_rvs_dataframes/training_flux.csv')
training_sigma_df = pd.read_csv('./data/gaia_rvs_dataframes/training_sigma.csv')

# =====================================================================================
# calculate relative flux weights of primary,secondary 

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
mass2teff = interp1d(mass_pm2013[valid_mass], teff_pm2013[valid_mass])

def flux_weights(teff1, teff2):
    # compute relative I band flux > flux ratio
    I1 = teff2Vmag(teff1) - teff2VminusI(teff1)
    I2 = teff2Vmag(teff2) - teff2VminusI(teff2)
    f1_over_f2 = 10**((I2-I1)/2.5)
    flux2_weight = 1/(1+f1_over_f2)
    flux1_weight = 1-flux2_weight
    return(flux1_weight, flux2_weight)

# =====================================================================================
#compute density of training set for a given set of model parameters
# training_label_df = pd.read_csv('./data/label_dataframes/training_labels.csv')
# training_labels = ['galah_teff', 'galah_logg','galah_feh', 'galah_alpha', 'galah_vbroad']
# training_data = training_label_df[training_labels].to_numpy()
training_data = recent_model_version.training_set_labels

# compute KDE from training set
training_density_kde = stats.gaussian_kde(training_data.T)

def training_density(cannon_params):
    density = training_density_kde(cannon_params)[0]
    return density

