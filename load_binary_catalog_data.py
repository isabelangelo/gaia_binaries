from astropy.table import Table
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astroquery
from astroquery.vizier import Vizier
import gaia

########### load data from catalogs ####################################################
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
galah_allstar_catalog_cleaned = galah_allstar_catalog.query('(snr_c3_iraf > 100) & (flag_sp == 0) \
& (flag_fe_h == 0) & (flag_alpha_fe == 0) & (ruwe_dr2 < 1.4)')
print(len(galah_allstar_catalog_cleaned), 'remaining after cleaning based on GALAH SNR + quality flags')

# reformat Gaia designation column
galah_gaia_xmatch['designation'] = [i.decode().replace('EDR3', 'DR3') for i in galah_gaia_xmatch['designation']]

########### crossmatch with GALAH allstars + Gaia and filter sample ############################
# merge GALAH binaries with GALAH star catalog
# to get labels for binaries when treated as single stars
galah_allstar_catalog_cols = ['sobject_id', 'star_id', 'teff', 'e_teff', 'logg', 'e_logg', 'fe_h', 'e_fe_h',\
                             'alpha_fe', 'e_alpha_fe','vbroad', 'e_vbroad','v_jk']
galah_gaia_xmatch_cols = ['sobject_id', 'designation']

galah_binaries_allstar = pd.merge(galah_binary_catalog, galah_allstar_catalog_cleaned[galah_allstar_catalog_cols], 
    left_on='spectID', right_on='sobject_id', validate='one_to_one')
galah_binaries_gaia = pd.merge(galah_binaries_allstar, galah_gaia_xmatch[galah_gaia_xmatch_cols], 
	on='sobject_id', validate='one_to_one')

print('{} binaries from Traven et al. (2020)'.format(len(galah_binary_catalog)))
print('{} found in GALAH allstar catalog'.format(len(galah_binaries_allstar)))

# remove duplicate rows based on designation,
# keep one observation (sobject_id) per unique gaia designation
galah_binaries_gaia = galah_binaries_gaia.drop_duplicates(subset='designation', keep='first')
galah_binaries_gaia = galah_binaries_gaia[galah_binaries_gaia.designation!=' ']
print('{} with unique Gaia designations'.format(len(galah_binaries_gaia)))


galah_binaries_gaia = galah_binaries_gaia.rename(
    columns={
    # GALAH star catalog columns
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
    "v_jk": "galah_vjk",
     # Traven 2020 columns
    'Teff1-50':'galah_teff1',
    'logg1-50':'galah_logg1',
    '__Fe_H_-50':'galah_feh12',
    'vbroad1-50':'galah_vbroad1',
    'Teff2-50':'galah_teff2',
    'logg2-50':'galah_logg2',
    'vbroad2-50':'galah_vbroad2',
    'RV1-50':'galah_rv1',
    'RV2-50':'galah_rv2'
    })

# require GALAH labels to be finite
training_set_labels = ['galah_teff', 'galah_logg', 'galah_feh', 'galah_alpha', 'galah_vbroad']
galah_binaries_gaia = galah_binaries_gaia.dropna(subset=training_set_labels)
print(len(galah_binaries_gaia), 'with finite training set labels')

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
galah_binaries_gaia = galah_binaries_gaia.query('galah_logg > 4 & galah_eteff<@emax_teff & galah_elogg<@emax_logg \
            & galah_efeh<@emax_feh & galah_ealpha<@emax_alpha\
            & galah_evbroad<@emax_vbroad')
print(len(galah_binaries_gaia), 'with logg>4, uncertainties < 2x median galah uncertainties')

########### upload to gaia to filter + download RVS spectra ##################################
# query to filter based on Gaia parameters + download RVS spectra
query = f"SELECT dr3.designation, galah.sobject_id, dr3.source_id \
FROM user_iangelo.galah_binaries_gaia as galah \
JOIN gaiadr3.gaia_source as dr3 \
    ON dr3.designation = galah.designation \
WHERE dr3.has_rvs = 'True' \
AND dr3.rvs_spec_sig_to_noise > 100 \
AND dr3.ruwe < 1.4"

# download gaia data and save to files
gaia.upload_table(galah_binaries_gaia, 'galah_binaries_gaia')
galah_binaries_gaia_results, galah_binaries_flux_df, galah_binaries_sigma_df = gaia.retrieve_data_and_labels(query)
print('{} with has_rvs = True, rvs snr > 100'.format(len(galah_binaries_gaia_results)))

print('saving flux, flux_err to .csv files')
gaia.write_labels_to_file(galah_binaries_gaia_results, 'galah_binaries')
gaia.write_flux_data_to_csv(galah_binaries_flux_df, galah_binaries_sigma_df, 'galah_binaries')


# the revised quality cuts are in the notion document
# note: the GALAH quality flags remove most of the binaries (sample goes from 11,000 to 200)
# even just having the GALAH snr cut removes most of them, but I'm going to keep this cut for now.











