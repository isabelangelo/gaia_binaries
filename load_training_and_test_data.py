# NOTE: before running this code, you need to delete galah_stars_gaia table from the gaia archive online
from astropy.table import Table
from astropy.io import fits
import numpy as np
import pandas as pd
from astroquery.vizier import Vizier
import gaia

########### load data from catalogs ################################################
# GALAH DR3 main catalog
galah_allstar_catalog = Table.read('./data/galah_catalogs/GALAH_DR3_main_allstar_v2.fits', format='fits').to_pandas()
# GALAH Gaia crossmatch for all spectra 
galah_gaia_xmatch = Table.read('./data/galah_catalogs/GALAH_DR3_VAC_GaiaEDR3_v2.fits', format='fits').to_pandas()
# GALAH binary catalog from Traven (2020)
catalogs = Vizier.find_catalogs('Traven')
catalogs = {k: v for k, v in catalogs.items() if 'J/A+A/638/A145' in k}
Vizier.ROW_LIMIT = -1
catalogs = Vizier.get_catalogs(catalogs.keys())
galah_binary_catalog = catalogs[0].to_pandas()

# following GALAH website best practices for clean star catalog
print(len(galah_allstar_catalog), 'GALAH stars in allstar catalog')
galah_allstar_catalog_cleaned = galah_allstar_catalog.query('(snr_c3_iraf > 30) & (flag_sp == 0) \
& (flag_fe_h == 0) & (flag_alpha_fe == 0)')
print(len(galah_allstar_catalog_cleaned), 'remaining after cleaning based on GALAH SNR + quality flags')

# reformat Gaia designation column
galah_gaia_xmatch['designation'] = [i.decode().replace('EDR3', 'DR3') for i in galah_gaia_xmatch['designation']]

########### crossmatch GALAH/gaia and filter sample ##################################
# find GALAH stars in Gaia xmatch table
galah_allstar_catalog_cols = ['sobject_id', 'star_id', 'teff', 'e_teff', 'logg', 'e_logg', 'fe_h', 'e_fe_h',\
                             'alpha_fe', 'e_alpha_fe','vbroad', 'e_vbroad','v_jk']
galah_gaia_xmatch_cols = ['sobject_id', 'designation']

# note: merge on sobject_id based on GALAH website
# designations matched on multiple sobject IDs are multiple observations of the same object
# keep observation (sobj_id) per designation (this is the default in drop_duplicates())
galah_stars_gaia = pd.merge(galah_gaia_xmatch[galah_gaia_xmatch_cols], 
	galah_allstar_catalog_cleaned[galah_allstar_catalog_cols], on='sobject_id')
galah_stars_gaia = galah_stars_gaia.drop_duplicates(subset='designation', keep='first')
galah_stars_gaia = galah_stars_gaia[galah_stars_gaia.designation!=' ']
print(len(galah_stars_gaia), 'stars with unique gaia designations from GALAH/gaia crossmatch')

# save relevant parameters, write to file
galah_stars_gaia = galah_stars_gaia.rename(
    columns={
    "teff": "galah_teff", 
    "e_teff": "galah_eteff",
    "logg":"galah_logg",
    "e_logg":"galah_elogg",
    "fe_h": "galah_feh", 
    "e_fe_h": "galah_efeh",
    "alpha_fe": "galah_alpha",
    "e_alpha_fe": "galah_ealpha",
    "vbroad":"galah_vbroad",
    "e_vbroad": "galah_evbroad",
    "v_jk": "galah_vjk"
    })

# require training set labels to be finite
training_set_labels = ['galah_teff', 'galah_logg', 'galah_feh', 'galah_alpha', 'galah_vbroad']
galah_stars_gaia = galah_stars_gaia.dropna(subset=training_set_labels)
print(len(galah_stars_gaia), 'with finite training set labels')

# filters to remove stars with label uncertainties >2*median GALAH uncertainty
def emax(colname):
	return 2*np.nanmedian(galah_stars_gaia[colname])

emax_teff = emax('galah_eteff')
emax_logg = emax('galah_elogg')
emax_feh = emax('galah_efeh')
emax_alpha = emax('galah_ealpha')
emax_vbroad = emax('galah_evbroad')
galah_stars_gaia = galah_stars_gaia.query('galah_logg > 4 & galah_eteff<@emax_teff & galah_elogg<@emax_logg \
            & galah_efeh<@emax_feh & galah_ealpha<@emax_alpha\
            & galah_evbroad<@emax_vbroad')
print(len(galah_stars_gaia), 'with logg>4, uncertainties < 2x median galah uncertainties')

# remove known binaries from training set
# note: using binary galah IDs from original vizier file yielded identical results
binary_galah_ids = galah_binary_catalog.spectID.to_numpy()
binary_idx_to_remove = []
for i in range(len(galah_stars_gaia)):
    row = galah_stars_gaia.iloc[i]
    if row.sobject_id in binary_galah_ids:
        binary_idx_to_remove.append(i)
galah_stars_gaia = galah_stars_gaia.drop(galah_stars_gaia.index[binary_idx_to_remove])
print(len(galah_stars_gaia), 'remaining after removing binaries from Traven et al. 2020')

########### upload to gaia to filter + download RVS spectra ##################################
# query to filter based on Gaia parameters + download RVS spectra
query = f"SELECT dr3.designation, galah.sobject_id, dr3.source_id, \
galah.galah_teff, galah.galah_eteff, galah.galah_logg, galah.galah_elogg, \
galah.galah_feh, galah.galah_efeh, galah.galah_alpha, galah.galah_ealpha, \
galah.galah_vbroad, galah.galah_evbroad, dr3.rvs_spec_sig_to_noise, \
dr3.ra, dr3.dec \
FROM user_iangelo.galah_stars_gaia as galah \
JOIN gaiadr3.gaia_source as dr3 \
	ON dr3.designation = galah.designation \
WHERE dr3.has_rvs = 'True' \
AND dr3.rvs_spec_sig_to_noise > 50 \
AND dr3.non_single_star = 0"

# query gaia and download RVS spectra, save to dataframes
gaia.upload_table(galah_stars_gaia, 'galah_stars_gaia')
galah_stars_gaia_results, flux_df, sigma_df = gaia.retrieve_data_and_labels(query)

# split into training + test sets
np.random.seed(1234)
print('splitting into training + test sets')
source_id_list = [int(i) for i in flux_df.columns]
n_total = len(source_id_list)
n_training = int(0.8*n_total)
n_test = n_total - n_training
training_source_ids = np.random.choice(source_id_list, size=n_training, replace=False)
test_source_ids = np.array(list((set(source_id_list) - set(training_source_ids))))

# write training + test set labels to .csv files
training_label_df = galah_stars_gaia_results[galah_stars_gaia_results['source_id'].isin(training_source_ids)]
test_label_df = galah_stars_gaia_results[galah_stars_gaia_results['source_id'].isin(test_source_ids)]
training_label_filename = './data/label_dataframes/training_labels.csv'
test_label_filename = './data/label_dataframes/test_labels.csv'
training_label_df.to_csv(training_label_filename)
test_label_df.to_csv(test_label_filename)
print('training set labels saved to {}'.format(training_label_filename))
print('test set labels saved to {}'.format(test_label_filename))

# write training + test set flux, sigma to .csv files
flux_sigma_df_path = './data/gaia_rvs_dataframes/'
training_flux_df_filename = flux_sigma_df_path + 'training_flux.csv'
test_flux_df_filename = flux_sigma_df_path + 'test_flux.csv'
training_sigma_df_filename = flux_sigma_df_path + 'training_sigma.csv'
test_sigma_df_filename = flux_sigma_df_path + 'test_sigma.csv'
flux_df[training_source_ids].to_csv(training_flux_df_filename)
flux_df[test_source_ids].to_csv(test_flux_df_filename)
sigma_df[training_source_ids].to_csv(training_sigma_df_filename)
sigma_df[test_source_ids].to_csv(test_sigma_df_filename)
print('training set flux, sigma dataframe saved to:\n{}\n{}'.format(training_flux_df_filename, training_sigma_df_filename))
print('test set flux, sigma dataframe saved to:\n{}\n{}'.format(test_flux_df_filename, test_sigma_df_filename))

# write training set flux, ivar to fits files for the cannon
training_flux_arr = flux_df[training_source_ids].to_numpy().T
training_sigma_arr = sigma_df[training_source_ids].to_numpy().T
training_flux_arr_filename = './data/cannon_training_data/training_flux.fits'
training_sigma_arr_filename = './data/cannon_training_data/training_sigma.fits'
fits.HDUList([fits.PrimaryHDU(training_flux_arr)]).writeto(training_flux_arr_filename, overwrite=True)
fits.HDUList([fits.PrimaryHDU(training_sigma_arr)]).writeto(training_sigma_arr_filename, overwrite=True)
print('training set flux array saved to {}'.format(training_flux_arr_filename))
print('training set sigma array saved to {}'.format(training_sigma_arr_filename))







