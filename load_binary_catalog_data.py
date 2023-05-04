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

# reformat Gaia designation column
galah_gaia_xmatch['designation'] = [i.decode().replace('EDR3', 'DR3') for i in galah_gaia_xmatch['designation']]


########### crossmatch with GALAH allstars + Gaia and filter sample ############################
# merge GALAH binaries with GALAH star catalog
# to get labels for binaries when treated as single stars
galah_allstar_catalog_cols = ['sobject_id', 'star_id', 'teff', 'e_teff', 'logg', 'e_logg', 'fe_h', 'e_fe_h',\
                             'alpha_fe', 'e_alpha_fe','vbroad', 'e_vbroad','v_jk']
galah_gaia_xmatch_cols = ['sobject_id', 'designation']
galah_binaries_allstar = pd.merge(galah_binary_catalog, galah_allstar_catalog[galah_allstar_catalog_cols], 
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

# require logg > 4
galah_binaries_gaia = galah_binaries_gaia.query('galah_logg > 4')
print('{} with galah logg > 4'.format(len(galah_binaries_gaia)))

########### upload to gaia to filter + download RVS spectra ##################################
# query to filter based on Gaia parameters + download RVS spectra
query = f"SELECT dr3.designation, galah.sobject_id, dr3.source_id \
FROM user_iangelo.galah_binaries_gaia as galah \
JOIN gaiadr3.gaia_source as dr3 \
    ON dr3.designation = galah.designation \
WHERE dr3.has_rvs = 'True' \
AND dr3.rvs_spec_sig_to_noise > 50"

# download gaia data and save to files
gaia.upload_table(galah_binaries_gaia, 'galah_binaries_gaia')
galah_binaries_gaia_results, galah_binaries_flux_df, galah_binaries_sigma_df = gaia.retrieve_data_and_labels(query)
print('{} with has_rvs = True, rvs snr > 50'.format(len(galah_binaries_gaia_results)))

print('saving flux, flux_err to .csv files')
gaia.write_labels_to_file(galah_binaries_gaia, 'galah_binaries')
gaia.write_flux_data_to_csv(galah_binaries_flux_df, galah_binaries_sigma_df, 'galah_binaries')


# next: I think I can add the other catalogs here...
# but they just query existing catalogs
# pw2020_binaries_filtered.csv - this comes from code to filter, although I do think it's gone...
# I'll have to look at notes from the Gaia fete.
# eb2018_binaries_filtered.csv
# where are these from? I can check the other branch/download from gaia/look at my notes










