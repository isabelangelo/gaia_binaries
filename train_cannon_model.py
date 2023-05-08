import numpy as np 
from astropy.table import Table
from astropy.io import fits
import thecannon as tc

# load wavelength of flux values
w_interp_to = fits.open('./data/cannon_training_data/gaia_rvs_wavelength.fits')[0].data[20:-20]

# Load the table containing the training set labels
training_set = Table.read('./data/label_dataframes/training_labels.csv', format='csv')
training_set = training_set['galah_teff', 'galah_logg','galah_feh', 'galah_alpha', 'galah_vbroad']

# load flux, clip of ends with nan values
normalized_flux_nan = fits.open('./data/cannon_training_data/training_flux.fits')[0].data[:, 20:-20]
normalized_sigma_nan = fits.open('./data/cannon_training_data/training_sigma.fits')[0].data[:, 20:-20]

# interpolate fluxes with nan values, set errors to 1
normalized_flux = normalized_flux_nan.copy()
for i in range(normalized_flux_nan.shape[0]):
    flux = normalized_flux_nan[i]
    finite_idx = ~np.isnan(flux)
    if np.sum(finite_idx) != len(flux):
        flux_interp = np.interp(w_interp_to, w_interp_to[finite_idx], flux[finite_idx])
        normalized_flux[i] = flux_interp

normalized_sigma = np.nan_to_num(normalized_sigma_nan, nan=1)
normalized_ivar = 1/normalized_sigma**2

# Create a vectorizer that defines our model form.
vectorizer = tc.vectorizer.PolynomialVectorizer(('galah_teff', 'galah_logg','galah_feh', 'galah_alpha', 'galah_vbroad'), 2)

# Create the model that will run in parallel using all available cores.
model = tc.CannonModel(training_set, normalized_flux, normalized_ivar,
                       vectorizer=vectorizer)

# train model
model_filename = './data/cannon_models/galah_labels_5para_highSNR_cleaned_v3.model'
model.train()
print('finished training cannon model')
model.write(model_filename)
print('model written to {}'.format(model_filename))
