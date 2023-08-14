"""
trains a cannon model and generates validation plots.
"""
import numpy as np 
import cannon_model_diagnostics
from astropy.table import Table
from astropy.io import fits
import thecannon as tc
import pandas as pd
import os

# path to save model files to, 
# should be descriptive of current model to be trained
model_fileroot = 'gaia_rvs_model'

################# define training set + get labels ###################################################
# define training set labels
training_labels = ['galah_teff', 'galah_logg','galah_feh', 'galah_alpha', 'galah_vbroad']

# Load the table containing the training set labels
training_set_table = Table.read('./data/label_dataframes/training_labels.csv', format='csv')
training_set = training_set_table[training_labels]

# normalized_sigma = np.nan_to_num(normalized_sigma_nan, nan=1)
normalized_flux = fits.open('./data/cannon_training_data/training_flux.fits')[0].data
normalized_sigma = fits.open('./data/cannon_training_data/training_sigma.fits')[0].data
normalized_ivar = 1/normalized_sigma**2


# Create a vectorizer that defines our model form.
vectorizer = tc.vectorizer.PolynomialVectorizer(training_labels, 2)

# Create the model that will run in parallel using all available cores.
model = tc.CannonModel(training_set, normalized_flux, normalized_ivar,
                       vectorizer=vectorizer, regularization=None)

# train model
model_filename = './data/cannon_models/' + model_fileroot + '.model'
model.train()
print('finished training cannon model')
model.write(model_filename, include_training_set_spectra=True)
print('model written to {}'.format(model_filename))

# print training set GALAH label errors
print('average GALAH label uncertainties in training set:')
for label in training_labels:
	print('{}: {}'.format(
		label,
		np.mean(training_set_table[label.replace('_', '_e')])))


# NOTE: I moved this section to cannon_model_diagnostics

# # save diagnostic plots
# model_figure_path = './data/cannon_models/'+model_fileroot+'_figures/'
# os.mkdir(model_figure_path)

# # plot histograms of training + test sets
# test_set = pd.read_csv('./data/abel_dataframes/test_labels.csv')
# training_histogram_filename = model_figure_path + 'training_set_plot.png'
# cannon_model_diagnostics.plot_training_set(
# 	training_set.to_pandas(), 
# 	test_set, 
# 	training_histogram_filename)
# print('training set histrogram saved to {}'.format(training_histogram_filename))

# # training set parameter space corner plot for 3 test spectra
# example_top_filename = model_figure_path + 'example_spec_top_panel.png'
# cannon_model_diagnostics.plot_example_spec_top_panel(
# 	training_set.to_pandas(), 
# 	example_top_filename)
# print('top panel of example spectrum plot saved to {}'.format(example_top_filename))

# # cannon model fits for 3 test spectra
# flux_df = pd.read_csv('./data/gaia_rvs_dataframes/training_flux.csv')
# sigma_df = pd.read_csv('./data/gaia_rvs_dataframes/training_sigma.csv')
# example_bottom_filename = model_figure_path +  'example_spec_bottom_panel.png'
# cannon_model_diagnostics.plot_example_spec_bottom_panel(
# 	training_set_table.to_pandas(),
# 	flux_df,
# 	sigma_df,
# 	model,
# 	example_bottom_filename)
# print('bottom panel of example spectrum plot saved to {}'.format(example_bottom_filename))

# # diagnostic plots from the cannon code
# theta_figure = tc.plot.theta(model)
# theta_figure.savefig(model_figure_path + 'theta.png', dpi=300)
# print('theta plot saved to {}'.format(model_figure_path + 'theta.png'))

# scatter_figure = tc.plot.scatter(model)
# scatter_figure.savefig(model_figure_path + 'scatter.png', dpi=300)
# print('pixel scatter plot saved to {}'.format(model_figure_path + 'scatter.png'))

# cannon_model_diagnostics.plot_one_to_one(
# 	training_set_table.to_pandas(),
# 	flux_df,
# 	sigma_df,
# 	model_figure_path + 'one_to_one_training.png',
# 	path_to_save_labels = model_fileroot+'_training_labels')
# print('one to one plot saved to {}'.format(model_figure_path + 'one_to_one.png'))

# test_label_df = pd.read_csv('./data/label_dataframes/test_labels.csv')
# test_flux_df = pd.read_csv('./data/gaia_rvs_dataframes/test_flux.csv')
# test_sigma_df = pd.read_csv('./data/gaia_rvs_dataframes/test_sigma.csv')
# cannon_model_diagnostics.plot_one_to_one(
#     test_label_df, 
#     test_flux_df, 
#     test_sigma_df,
#     model_figure_path + 'one_to_one_test.png')

