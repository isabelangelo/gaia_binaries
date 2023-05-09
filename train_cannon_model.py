import numpy as np 
import plot
from astropy.table import Table
from astropy.io import fits
import thecannon as tc
import pandas as pd
import os

# define training set labels
training_labels = ['galah_teff', 'galah_logg','galah_feh', 'galah_alpha', 'galah_vbroad']

# Load the table containing the training set labels
training_set_table = Table.read('./data/galah_label_dataframes/training_labels.csv', format='csv')
training_set = training_set_table[training_labels]

# load wavelength of flux values
w_interp_to = fits.open('./data/cannon_training_data/gaia_rvs_wavelength.fits')[0].data[20:-20]

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
vectorizer = tc.vectorizer.PolynomialVectorizer(training_labels, 2)

# Create the model that will run in parallel using all available cores.
model = tc.CannonModel(training_set, normalized_flux, normalized_ivar,
                       vectorizer=vectorizer)

# train model
model_fileroot = 'galah_labels_5para_highSNR_cleaned_v3'
model_filename = './data/cannon_models/' + model_fileroot + '.model'
model.train()
print('finished training cannon model')
model.write(model_filename)
print('model written to {}'.format(model_filename))

# print training set GALAH label errors
print('average GALAH label uncertainties in training set:')
for label in training_set_labels:
	print('{}: {}'.format(
		label,
		np.mean(training_set_table[label.replace('_', '_e')])))

# save diagnostic plots
model_figure_path = './data/cannon_models/'+model_fileroot+'_figures/'
os.mkdir(model_figure_path)

# commenting this out for now since it's pretty redundant with the one-to-one plot
# but it might be useful if I need a plot of just the training set later on
# test_set = pd.read_csv('./data/galah_label_dataframes/test_labels.csv')
# training_histogram_filename = model_figure_path + 'training_set_plot.png'
# plot.plot_training_set(
# 	training_set.to_pandas(), 
# 	test_set, 
# 	training_histogram_filename)
# print('training set histrogram saved to {}'.format(training_histogram_filename))

# example_top_filename = model_figure_path + 'example_spec_top_panel.png'
# plot.plot_example_spec_top_panel(
# 	training_set.to_pandas(), 
# 	example_top_filename)
# print('top panel of example spectrum plot saved to {}'.format(example_top_filename))

# flux_df = pd.read_csv('./data/gaia_rvs_dataframes/training_flux.csv')
# sigma_df = pd.read_csv('./data/gaia_rvs_dataframes/training_sigma.csv')
# example_bottom_filename = model_figure_path +  'example_spec_bottom_panel.png'
# plot.plot_example_spec_bottom_panel(
# 	training_set_table.to_pandas(),
# 	flux_df,
# 	sigma_df,
# 	model,
# 	example_bottom_filename)
# print('bottom panel of example spectrum plot saved to {}'.format(example_bottom_filename))

theta_figure = tc.plot.theta(model)
theta_figure.savefig(model_figure_path + 'theta.png', dpi=300)
print('theta plot saved to {}'.format(model_figure_path + 'theta.png'))

scatter_figure = tc.plot.scatter(model)
scatter_figure.savefig(model_figure_path + 'scatter.png', dpi=300)
print('pixel scatter plot saved to {}'.format(model_figure_path + 'scatter.png'))

plot.plot_one_to_one(
	training_set_table.to_pandas(),
	flux_df,
	sigma_df,
	model,
	model_figure_path + 'one_to_one.png',
	path_to_save_labels = 'training_cannon_labels_v3',
	labels_to_plot = ['galah_teff', 'galah_logg', 'galah_feh', 'galah_alpha', 'galah_vbroad'])
print('one to one plot saved to {}'.format(model_figure_path + 'one_to_one.png'))


# for now, I'm only making the cannon plots (theta, scatter, one_to_one)

# then when I add the new labels, I should rename the columns
# galah_X so that the rest of the code works etc.
# I'll need to change L12-14
# and I need both training_set_table and training_set to be changed


