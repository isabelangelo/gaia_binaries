import numpy as np 
import pandas as pd
from astropy.table import Table
from astropy.io import fits

# Load the table containing the training set labels
training_set_df = pd.read_csv('./GALAH_csv_files/GALAH_training_set.csv')
training_set_df_cannon_labels = training_set_df[['teff_gspphot','logg_gspphot', 'mh_gspphot', 'vbroad']]
training_set = Table.from_pandas(training_set_df_cannon_labels) # convert to astropy table

# load the spectra flux + flux errors
f_start = './GALAH_training_set_RVS_spectra/RVS-'
f_end = '.csv'
# flux
print('loading flux for {} spectra'.format(len(training_set_df)))
normalized_flux_nan = np.array([pd.read_csv(f_start+i+f_end).flux.to_numpy() for i in training_set_df.dr3_designation])

# inverse variance (gaia reports error= std dev = sqrt(variance))
print('loading flux errors for {} spectra'.format(len(training_set_df)))
normalized_sigma = np.array([pd.read_csv(f_start+i+f_end).flux_error.to_numpy() for i in training_set_df.dr3_designation])
normalized_ivar_nan = 1./(normalized_sigma**2.)# convert to ivar

# write to fits files
print('writing flux, ivar to fits files...')
flux_hdu = fits.PrimaryHDU(normalized_flux_nan)
ivar_hdu = fits.PrimaryHDU(normalized_ivar_nan)

flux_filename = 'cannon_flux_data.fits'
fits.HDUList([flux_hdu]).writeto(flux_filename)
print('flux saved to {}'.format(flux_filename))

ivar_filename = 'cannon_ivar_data.fits'
fits.HDUList([ivar_hdu]).writeto(ivar_filename)
print('ivar saved to {}'.format(ivar_filename))