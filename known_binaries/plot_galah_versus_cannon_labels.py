
from astropy.table import Table
import pandas as pd
import thecannon as tc
from astropy.io import fits
import matplotlib.pyplot as plt

# load cannon model
model = tc.CannonModel.read('../train_cannon_model/cannon_models/galah_labels_5para_highSNR_precise_labels.model')

# load flux data, training + test sets, cannon model
w = fits.open('../train_cannon_model/data_files/gaia_rvs_wavelength.fits')[0].data[20:-20]

# load star data
test_set = pd.read_csv('../train_cannon_model/data_files/cannon_test_set.csv')
training_set = pd.read_csv('../train_cannon_model/data_files/cannon_training_set.csv')
flux_data = pd.read_csv('../train_cannon_model/data_files/flux_data.csv')
flux_err_data = pd.read_csv('../train_cannon_model/data_files/flux_err_data.csv')

# load binary data (need to merge to get original GALAH labels)
test_binaries = pd.read_csv('../process_GALAH_data/GALAH_data_tables/GALAH_binaries_filtered.csv')

binary_flux_data = pd.read_csv('../known_binaries/data_files/flux_data_binaries.csv')
binary_flux_err_data = pd.read_csv('../known_binaries/data_files/flux_err_data_binaries.csv')

def get_labels(label_df, flux_data_df, flux_err_data_df):
    """
    function to store pre-determined and cannon labels for comparison

    Args:
        label_df (pandas.Dataframe) : dataframe containing pre-determined labels
        flux_data_df (pandas.Dataframe) : dataframe with fluxes for each sobject_id
        flux_err_data_df (pandas.Dataframe) : dataframe with flux errors for each sobject_id

    Returns:
        cannon_label_df (pandas.Dataframe) : dataframe with pre-determined and cannon labels for all objects
        in cannon_label_df
    """
    # lists to store labels
    teff_galah, teff_cannon = [], []
    logg_galah, logg_cannon = [], []
    feh_galah, feh_cannon = [], []
    alpha_galah, alpha_cannon = [], []
    vbroad_galah, vbroad_cannon = [], []
    non_single_star = []

    # iterate over each object
    for source_id in label_df.source_id.to_numpy():

        # store galah labels
        row = label_df.loc[label_df.source_id==source_id]
        teff_galah.append(row['galah_teff'].to_numpy()[0])
        logg_galah.append(row['galah_logg'].to_numpy()[0])
        feh_galah.append(row['galah_feh'].to_numpy()[0])
        alpha_galah.append(row['galah_alpha_fe'].to_numpy()[0])
        vbroad_galah.append(row['galah_vbroad'].to_numpy()[0])
        non_single_star.append(row['non_single_star'].to_numpy()[0])
        
        # fit cannon model
        flux = flux_data_df[str(source_id)].to_numpy()[20:-20]
        ivar = 1/flux_err_data_df[str(source_id)].to_numpy()[20:-20]**2
        result = model.test(flux, ivar)
        teff_fit, logg_fit, feh_fit, alpha_fit, vbroad_fit = result[0][0]
        
        # store cannon labels
        teff_cannon.append(teff_fit)
        logg_cannon.append(logg_fit)
        feh_cannon.append(feh_fit)
        alpha_cannon.append(alpha_fit)
        vbroad_cannon.append(vbroad_fit)

    # store data in dataframe
    data = {'teff_galah':teff_galah, 
            'logg_galah':logg_galah,
            'feh_galah': feh_galah,
            'alpha_galah':alpha_galah,
            'vbroad_galah':vbroad_galah,
            'teff_cannon':teff_cannon, 
            'logg_cannon':logg_cannon,
            'feh_cannon': feh_cannon,
            'alpha_cannon':alpha_cannon,
            'vbroad_cannon':vbroad_cannon,
            'non_single_star': non_single_star}

    cannon_label_df = pd.DataFrame(data)
    return cannon_label_df

