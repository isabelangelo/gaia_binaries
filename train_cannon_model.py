import numpy as np 
import plot
from astropy.table import Table
from astropy.io import fits
import thecannon as tc
import pandas as pd
import os

# load wavelength of flux values
w_interp_to = fits.open('./data/cannon_training_data/gaia_rvs_wavelength.fits')[0].data[20:-20]

# Load the table containing the training set labels
training_set_table = Table.read('./data/galah_label_dataframes/training_labels.csv', format='csv')
training_set = training_set_table['galah_teff', 'galah_logg','galah_feh', 'galah_alpha', 'galah_vbroad']

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
model_fileroot = 'galah_labels_5para_highSNR_cleaned_v3'
model_filename = './data/cannon_models/' + model_fileroot + '.model'
model.train()
print('finished training cannon model')
# model.write(model_filename)
print('model written to {}'.format(model_filename))

# save diagnostic plots
figure_path = './data/cannon_models/'+model_fileroot+'_figures/'
# os.mkdir(figure_path)

test_set = pd.read_csv('./data/galah_label_dataframes/test_labels.csv')
plot.plot_training_set(
	training_set.to_pandas(), 
	test_set, 
	figure_path + 'training_set_plot.png')

plot.plot_example_spec_top_panel(
	training_set.to_pandas(), 
	figure_path + 'example_spec_top_panel.png')

flux_df = pd.read_csv('./data/gaia_rvs_dataframes/training_flux.csv')
sigma_df = pd.read_csv('./data/gaia_rvs_dataframes/training_sigma.csv')
plot.plot_example_spec_bottom_panel(
	training_set_table.to_pandas(),
	flux_df,
	sigma_df,
	model,
	figure_path +  'example_spec_bottom_panel.png')





