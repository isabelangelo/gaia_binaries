"""
trains a cannon model and generates validation plots.
"""
import numpy as np 
from astropy.table import Table
from astropy.io import fits
from cannon_model_diagnostics import *
import thecannon as tc
import pandas as pd

# path to save model files to, 
# should be descriptive of current model to be trained
model_fileroot = 'gaia_rvs_model'

################# train Cannon model ###################################################
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
model_0 = tc.CannonModel(training_set, normalized_flux, normalized_ivar,
                       vectorizer=vectorizer, regularization=None)

# train initial model and write to file for reference
model_0.train()
update_s2_emp(model_0)
print('finished training first iteration of cannon model')
model_filename_initial = './data/cannon_models/' + model_fileroot + '_initial.model'
model_0.write(model_filename_initial, include_training_set_spectra=True)

# clean until the model finds zero binaries in its training set
model_n = model_0 
model_n_plus_1 = clean(model_0) 
n_iter = 1
n_binaries = model_n.training_set_labels.shape[0] - model_n_plus_1.training_set_labels.shape[0]
print('iteration {}: {} binaries found, training set size = {}'.format(
    n_iter, 
    n_binaries, 
    model_n_plus_1.training_set_labels.shape[0]))

while n_binaries>0:
    model_n = model_n_plus_1
    model_n_plus_1 = clean(model_n_plus_1)
    n_binaries = model_n.training_set_labels.shape[0] - model_n_plus_1.training_set_labels.shape[0]
    n_iter += 1
    print('iteration {}: {} binaries found, training set size = {}'.format(
        n_iter, 
        n_binaries, 
        model_n_plus_1.training_set_labels.shape[0]))

# save cleaned model
model_cleaned = model_n_plus_1
model_filename_cleaned = './data/cannon_models/' + model_fileroot + '_cleaned.model'
model_cleaned.write(model_filename_cleaned, include_training_set_spectra=True)
print('cleaned model written to {}'.format(model_filename_cleaned))

################# update training data and save to files ###################################################

# save the full training set labels + flux for the cleaned model
training_set_table_df = training_set_table.to_pandas() # need pandas version to query
training_flux_df = pd.read_csv('./data/gaia_rvs_dataframes/training_flux.csv')
training_sigma_df = pd.read_csv('./data/gaia_rvs_dataframes/training_sigma.csv')

# determine rows from original training set that are in cleaned training set
cleaned_model_rows = []
for i in range(len(training_set_table_df)):
    row = training_set_table_df.iloc[i][training_labels]
    if row.values in model_cleaned.training_set_labels:
        cleaned_model_rows.append(i)

# index training set data at indices of objects in cleaned dataset
training_labels_cleaned = training_set_table_df.iloc[cleaned_model_rows]
training_flux_df_cleaned = training_flux_df[[str(i) for i in training_labels_cleaned.source_id]]
training_sigma_df_cleaned = training_sigma_df[[str(i) for i in training_labels_cleaned.source_id]]

# save data to files
training_labels_cleaned.to_csv('./data/label_dataframes/training_labels_cleaned.csv')
training_flux_df_cleaned.to_csv('./data/gaia_rvs_dataframes/training_flux_cleaned.csv')
training_sigma_df_cleaned.to_csv('./data/gaia_rvs_dataframes/training_sigma_cleaned.csv')

################# generate + save diagnostic plots ###################################################

# define path to save plots to
model_figure_path = './data/cannon_models/'+custom_model.recent_model_fileroot+'_figures/'
os.mkdir(model_figure_path)

# load data for plots
training_label_df_cleaned = pd.read_csv('./data/label_dataframes/training_labels_cleaned.csv')
training_flux_df_cleaned = pd.read_csv('./data/gaia_rvs_dataframes/training_flux_cleaned.csv')
training_sigma_df_cleaned = pd.read_csv('./data/gaia_rvs_dataframes/training_sigma_cleaned.csv')

# save one-to-one plot
plot_one_to_one(
    training_label_df_cleaned,
    training_flux_df_cleaned,
    training_sigma_df_cleaned,
    model_figure_path + 'one_to_one.png',
    path_to_save_labels = custom_model.recent_model_fileroot+'_training_labels')
print('one to one plot saved to {}'.format(model_figure_path + 'one_to_one.png'))

