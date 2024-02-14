"""
loads labels + RVS spectra for the Cannon training and test sets.
NOTE: before running this code, you need to 
delete galah_stars_gaia table from the gaia archive online
"""
from astropy.table import Table
import numpy as np
import pandas as pd
from astroquery.vizier import Vizier
import gaia

########### load data from catalogs ################################################
# GALAH DR3 main catalog
galah_catalog_path = './data/literature_data/survey_catalogs/'
galah_allstar_catalog = Table.read(galah_catalog_path + 'GALAH_DR3_main_allstar_v2.fits', 
    format='fits').to_pandas()
# GALAH Gaia crossmatch for all spectra 
galah_gaia_xmatch = Table.read(galah_catalog_path + 'GALAH_DR3_VAC_GaiaEDR3_v2.fits', 
    format='fits').to_pandas()
# GALAH binary catalog from Traven (2020)
catalogs = Vizier.find_catalogs('Traven')
catalogs = {k: v for k, v in catalogs.items() if 'J/A+A/638/A145' in k}
Vizier.ROW_LIMIT = -1
catalogs = Vizier.get_catalogs(catalogs.keys())
galah_binary_catalog = catalogs[0].to_pandas()

# following GALAH website best practices for clean star catalog
print(len(galah_allstar_catalog), 'GALAH stars in allstar catalog')
galah_allstar_catalog_cleaned = galah_allstar_catalog.query('(snr_c3_iraf > 100) & (flag_sp == 0) \
& (flag_fe_h == 0) & (flag_alpha_fe == 0) & (ruwe_dr2 < 1.4)')
print(len(galah_allstar_catalog_cleaned), 'remaining after cleaning based on GALAH SNR + quality flags')

# reformat Gaia designation column
galah_gaia_xmatch['designation'] = [i.decode().replace('EDR3', 'DR3') for i in galah_gaia_xmatch['designation']]


########### crossmatch GALAH/gaia and filter sample ##################################
# find GALAH stars in Gaia xmatch table
galah_allstar_catalog_cols = ['sobject_id', 'star_id', 'teff', 'e_teff', 'logg', 'e_logg', 'fe_h', 'e_fe_h',\
                             'alpha_fe', 'e_alpha_fe','vbroad', 'e_vbroad','v_jk', 'snr_c3_iraf']
galah_gaia_xmatch_cols = ['sobject_id', 'designation']

# note: merge on sobject_id based on GALAH website
# designations matched on multiple sobject IDs are multiple observations of the same object
# keep observation (sobj_id) per designation (this is the default in drop_duplicates())
galah_stars_gaia = pd.merge(
    galah_gaia_xmatch[galah_gaia_xmatch_cols], 
	galah_allstar_catalog_cleaned[galah_allstar_catalog_cols],
     on='sobject_id')
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
# and require GALAH logg>4
def emax(colname):
	return 2*np.nanmedian(galah_allstar_catalog[colname])
# using original column names to filter based on errors in GALAH allstars
emax_teff = emax('e_teff') 
emax_logg = emax('e_logg')
emax_feh = emax('e_fe_h')
emax_alpha = emax('e_alpha_fe')
emax_vbroad = emax('e_vbroad')

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
dr3.ra, dr3.dec, galah.snr_c3_iraf \
FROM user_iangelo.galah_stars_gaia as galah \
JOIN gaiadr3.gaia_source as dr3 \
	ON dr3.designation = galah.designation \
WHERE dr3.has_rvs = 'True' \
AND dr3.rvs_spec_sig_to_noise > 100 \
AND dr3.ruwe < 1.4 \
AND dr3.non_single_star = 0"

# query gaia and download RVS spectra, save to dataframes
gaia.upload_table(galah_stars_gaia, 'galah_stars_gaia')
galah_stars_gaia_results, flux_df, sigma_df = gaia.retrieve_data_and_labels(query)
print('{} with has_rvs = True, snr cuts, non_single_star = 0'.format(len(galah_stars_gaia_results)))
print('saving flux, flux_err to .csv files')

# write training set labels to .csv files
gaia.write_labels_to_file(galah_stars_gaia_results, 'training')

# write training set flux, sigma to .csv and .fits files
gaia.write_flux_data_to_csv(flux_df, sigma_df, 'training')
gaia.write_flux_data_to_fits(flux_df, sigma_df, 'training')


########### load high SNR sample for empriical scatter calculation ################################
# query to filter based on Gaia parameters + download RVS spectra
high_snr_query = f"SELECT TOP 100 dr3.source_id, dr3.rvs_spec_sig_to_noise \
FROM gaiadr3.gaia_source as dr3 \
WHERE dr3.has_rvs = 'True' \
AND dr3.logg_gspphot > 4 \
AND dr3.teff_gspphot > 4000 \
AND dr3.teff_gspphot < 7000 \
AND dr3.ruwe < 1.4 \
AND dr3.non_single_star = 0 \
ORDER BY dr3.rvs_spec_sig_to_noise DESC"

# query gaia and download RVS spectra, save to dataframes
_, high_snr_flux_df, high_snr_sigma_df = gaia.retrieve_data_and_labels(high_snr_query)

# write training set flux, sigma to .csv files
gaia.write_flux_data_to_csv(high_snr_flux_df, high_snr_sigma_df, 'high_snr')




