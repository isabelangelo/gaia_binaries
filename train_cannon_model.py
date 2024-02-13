"""
trains a cannon model and generates validation plots.
"""
import numpy as np 
from astropy.table import Table
from astropy.io import fits
import thecannon as tc
import gaia_spectrum
import pandas as pd

################# define variables and functions for training ###################################################

# path to save model files to, 
# should be descriptive of current model to be trained
model_fileroot = 'gaia_rvs_model'

# high SNR spectra for empirical s2 calculation 
high_snr_flux_df = pd.read_csv('./data/gaia_rvs_dataframes/high_snr_flux.csv')
high_snr_sigma_df = pd.read_csv('./data/gaia_rvs_dataframes/high_snr_sigma.csv')

def update_s2_emp(model):
    """
    function to compute empirical model scatter and update model.s2_emp parameter
    
    Args:
        model (tc.model.CannonModel) : Cannon model object to compute s2 for
    Returns:
        model (tc.model.CannonModel) : updated Cannon model object with s2_emp attribute
    """
    # compute |data - model| for high SNR targets
    s2_emp_arr = []
    for source_id in high_snr_flux_df.columns.to_numpy()[1:]:
        # load flux, sigma of target
        flux = high_snr_flux_df[source_id]
        sigma = high_snr_sigma_df[source_id]
        # mask calcium features for fit
        spec = gaia_spectrum.GaiaSpectrum(int(source_id), flux, sigma)
        # store per pixel |data-model|
        s2_emp_arr.append(abs(flux - spec.single_fit))
    # compute per pixel empirical s2, save as model attribute
    s2_emp_arr = np.array(s2_emp_arr).T
    model.s2_emp = np.mean(s2_emp_arr, axis=1)**2
    return model

def clean(model_iter_n):
    """
    function to clean (i.e. remove binaries) from training set and re-train
    binaries are identified as objects with delta_chisq>100, 
    used for iterative cleaning
    
    Args:
        model_iter_n (tc.model.CannonModel) : Cannon model object to clean
    Returns:
        model_iter_n_plus_1 (tc.model.CannonModel) : cleaned cannon model object
    """
    n_training_set_spectra = len(model_iter_n.training_set_labels)
    training_set_delta_chisq = np.zeros(n_training_set_spectra)
    
    # compute delta chisq for all objects in training set
    for spec_idx in range(n_training_set_spectra):
        spec_flux = model_iter_n.training_set_flux[spec_idx]
        spec_sigma = 1/np.sqrt(model_iter_n.training_set_ivar[spec_idx])
        spec_labels = model_iter_n.training_set_labels[spec_idx]
        spec = gaia_spectrum.GaiaSpectrum(
            None, 
            spec_flux, 
            spec_sigma, 
            model_to_use=model_iter_n)
        spec.compute_binary_detection_stats()
        training_set_delta_chisq[spec_idx] = spec.delta_chisq
        
    # make new training set with only objects labels as single stars   
    idx_to_keep = (training_set_delta_chisq < 100)
    training_set_iter_n_plus_1 = model_iter_n.training_set_labels[idx_to_keep]
    normalized_flux_iter_n_plus_1 = model_iter_n.training_set_flux[idx_to_keep]
    normalized_ivar_iter_n_plus_1 = model_iter_n.training_set_ivar[idx_to_keep]
    
    # re-train model
    model_iter_n_plus_1 = tc.CannonModel(
        training_set_iter_n_plus_1, 
        normalized_flux_iter_n_plus_1, 
        normalized_ivar_iter_n_plus_1,
        vectorizer=vectorizer, 
        regularization=None)
    model_iter_n_plus_1.train()

    # udpate empirical scatter for newest iteration
    update_s2_emp(model_iter_n_plus_1)
    return model_iter_n_plus_1

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

# # print training set GALAH label errors
# print('average GALAH label uncertainties in training set:')
# for i in range(len(training_labels)):
# 	label = training_labels[i]
# 	label_uncertainty = model_cleaned.training_set_labels.T[i].mean()
# 	print('{}: {}'.format(
# 		label,
# 		np.mean(training_set_table[label.replace('_', '_e')])))

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

