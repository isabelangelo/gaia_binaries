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
# load cannon models to use + wavelength
w = fits.open('./data/cannon_training_data/gaia_rvs_wavelength.fits')[0].data[20:-20]
recent_model_fileroot = 'gaia_rvs_model_cleaned'
recent_model_version = tc.CannonModel.read('./data/cannon_models/'+recent_model_fileroot+'.model')

# TEMPORARY set empirical s2 for chisq calculation
# NOTE: when I delete this, I also need to change the weights term in the 
# single star + binary residuals function
recent_model_version.s2_emp = fits.open('./empirical_model_s2.fits')[0].data

training_labels = ['galah_teff', 'galah_logg','galah_feh', 'galah_alpha', 'galah_vbroad']
training_set_table = Table.read('./data/label_dataframes/training_labels.csv', format='csv')
training_set = training_set_table[training_labels]

training_flux_df = pd.read_csv('./data/gaia_rvs_dataframes/training_flux.csv')
training_sigma_df = pd.read_csv('./data/gaia_rvs_dataframes/training_sigma.csv')

# =====================================================================================
# define training density function to use in custom model fitting, post-training
training_data = recent_model_version.training_set_labels
training_density_kde = stats.gaussian_kde(training_data.T)
def training_density(param):
    density = training_density_kde(param)[0]
    return density

# =====================================================================================
# define calcium mask
# narrow mask
ca_idx1 = np.where((w>849.5) & (w<850.5))[0]
ca_idx2 = np.where((w>854) & (w<855))[0]
ca_idx3 = np.where((w>866) & (w<867))[0]

# # broad mask
# ca_idx1 = np.where((w>849) & (w<851))[0]
# ca_idx2 = np.where((w>853.5) & (w<855.5))[0]
# ca_idx3 = np.where((w>865.5) & (w<867.5))[0]

# compute full mask from components
ca_mask = np.array(list(ca_idx1) + list(ca_idx2) + list(ca_idx3))

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

