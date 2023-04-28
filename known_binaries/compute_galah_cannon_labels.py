
from astropy.table import Table
import pandas as pd
import thecannon as tc
from astropy.io import fits
import matplotlib.pyplot as plt

# load cannon model
model = tc.CannonModel.read('../train_cannon_model/cannon_models/galah_labels_5para_highSNR_cleaned.model')

# load flux data, training + test sets, cannon model
w = fits.open('../train_cannon_model/data_files/gaia_rvs_wavelength.fits')[0].data[20:-20]

# # load star data
# test_set = pd.read_csv('../train_cannon_model/data_files/cannon_test_set.csv')
# training_set = pd.read_csv('../train_cannon_model/data_files/cannon_training_set.csv')
# flux_data = pd.read_csv('../train_cannon_model/data_files/flux_data.csv')
# flux_err_data = pd.read_csv('../train_cannon_model/data_files/flux_err_data.csv')

# # load binary data (need to merge to get original GALAH labels)
# test_binaries = pd.read_csv('../process_GALAH_data/GALAH_data_tables/GALAH_binaries_filtered.csv')
# binary_flux_data = pd.read_csv('../known_binaries/data_files/flux_data_binaries.csv')
# binary_flux_err_data = pd.read_csv('../known_binaries/data_files/flux_err_data_binaries.csv')

# load binary data from Price-Whelan 2020 sample
pw2020_binaries = pd.read_csv('./data_files/PriceWhelan2020_binaries_filtered.csv')
pw2020_flux_data = pd.read_csv('./data_files/flux_data_pw2020.csv')
pw2020_flux_err_data = pd.read_csv('./data_files/flux_data_pw2020.csv')
# remove binaries with SNR<50
flux_source_id_list_pw2020 = [int(i) for i in pw2020_flux_data.columns]
pw2020_binaries = pw2020_binaries[pw2020_binaries['source_id'].isin(flux_source_id_list_pw2020)]

# load binary data from El-Badry 2018 sample
eb2018_binaries = pd.read_csv('./data_files/ElBadry2018_binaries_filtered.csv')
eb2018_flux_data = pd.read_csv('./data_files/flux_data_eb2018.csv')
eb2018_flux_err_data = pd.read_csv('./data_files/flux_data_eb2018.csv')
# remove binaries with SNR<50
flux_source_id_list_eb2018 = [int(i) for i in eb2018_flux_data.columns]
eb2018_binaries = eb2018_binaries[eb2018_binaries['source_id'].isin(flux_source_id_list_eb2018)]

import pdb;pdb.set_trace()

def n(row_subset):
    return row_subset.to_numpy()[0]

def get_labels(label_df, flux_data_df, flux_err_data_df, inst='GALAH'):

    if inst=='GALAH':
        keys = ['sobject_id', 'galah_teff', 'galah_logg', 'galah_feh', 'galah_alpha_fe', 'galah_vbroad', \
        'non_single_star', 'rvs_spec_sig_to_noise', 'cannon_teff', 'cannon_logg', \
        'cannon_feh', 'cannon_alpha', 'cannon_vbroad', 'cannon_chi_sq', 'cannon_r_chi_sq']

    elif inst=='APOGEE':
        keys = ['apogee_id', 'source_id', 'apogee_teff', 'apogee_logg', \
        'apogee_feh', 'cannon_teff', 'cannon_logg', 'cannon_feh', 'cannon_alpha', \
        'cannon_vbroad', 'cannon_chi_sq', 'cannon_r_chi_sq']

    data = []

    # iterate over each object
    for source_id in label_df.source_id.to_numpy():

        # store galah labels
        row = label_df.loc[label_df.source_id==source_id]
        original_dataset_labels = row[keys[:-7]].to_numpy()

        # fit cannon model
        flux = flux_data_df[str(source_id)].to_numpy()[20:-20]
        ivar = 1/flux_err_data_df[str(source_id)].to_numpy()[20:-20]**2
        result = model.test(flux, ivar)
        teff_fit, logg_fit, feh_fit, alpha_fit, vbroad_fit = result[0][0]
        
        # store cannon labels
        cannon_labels = [teff_fit, logg_fit, feh_fit, alpha_fit, vbroad_fit, \
        result[2][0]['chi_sq'], result[2][0]['r_chi_sq']]

        # convert to dictionary
        values = original_dataset_labels.tolist()[0]+cannon_labels
        data.append(dict(zip(keys, values)))

    cannon_label_df = pd.DataFrame(data)

    return cannon_label_df


# generate dataframes with labels form GALAH + the Cannon

training_set_df = get_labels(training_set, flux_data, flux_err_data)
training_set_filename = '../process_GALAH_data/GALAH_data_tables/training_set_galah_cannon_labels.csv'
training_set_df.to_csv(training_set_filename)
print('training set labels saved to {}'.format(training_set_filename))

test_set_df = get_labels(test_set, flux_data, flux_err_data)
test_set_filename = '../process_GALAH_data/GALAH_data_tables/test_set_galah_cannon_labels.csv'
test_set_df.to_csv(test_set_filename)
print('test set labels saved to {}'.format(test_set_filename))

known_binary_df = get_labels(test_binaries, binary_flux_data, binary_flux_err_data)
known_binary_filename = '../process_GALAH_data/GALAH_data_tables/known_binary_galah_cannon_labels.csv'
known_binary_df.to_csv(known_binary_filename)
print('known binary labels saved to {}'.format(known_binary_filename))

# generate dataframes with labels from APOGEE + the Cannon

pw2020_binary_df = get_labels(pw2020_binaries, pw2020_flux_data, pw2020_flux_err_data, inst='APOGEE')
pw2020_binary_filename = './data_files/pw2020_binary_apogee_cannon_labels.csv'
pw2020_binary_df.to_csv(pw2020_binary_filename)
print('Price-Whelan 2020 binary labels saved to {}'.format(pw2020_binary_filename))

eb2018_binary_df = get_labels(eb2018_binaries, eb2018_flux_data, eb2018_flux_err_data, inst='APOGEE')
eb2018_binary_filename = './data_files/eb2018_binary_apogee_cannon_labels.csv'
eb2018_binary_df.to_csv(eb2018_binary_filename)
print('El-Badry 2018 binary labels saved to {}'.format(eb2018_binary_filename))












