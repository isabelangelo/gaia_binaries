"""
loads RVS spectra + labels for control sample from GALAH    
these are stars with similar filters to the vetted binaries from Traven et al. 2020,
except there are no filters with a preference for or against binarity
"""
from astropy.table import Table
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astroquery
from astroquery.vizier import Vizier
import gaia

########### load data from catalogs ################################################
# GALAH DR3 main catalog
galah_catalog_path = './data/literature_data/galah_catalogs/'
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

# print number of stars in allstar catalog
print(len(galah_allstar_catalog), 'GALAH stars in allstar catalog')
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
galah_control_gaia = pd.merge(galah_gaia_xmatch[galah_gaia_xmatch_cols], 
	galah_allstar_catalog[galah_allstar_catalog_cols], on='sobject_id')
galah_control_gaia = galah_control_gaia.drop_duplicates(subset='designation', keep='first')
galah_control_gaia = galah_control_gaia[galah_control_gaia.designation!=' ']
print(len(galah_control_gaia), 'stars with unique gaia designations from GALAH/gaia crossmatch')

# save relevant parameters, write to file
galah_control_gaia = galah_control_gaia.rename(
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

# filters to require GALAH logg>4
galah_control_gaia = galah_control_gaia.query('galah_logg > 4')
print(len(galah_control_gaia), 'with logg>4')

# remove known binaries from training set
# note: using binary galah IDs from original vizier file yielded identical results
binary_galah_ids = galah_binary_catalog.spectID.to_numpy()
binary_idx_to_remove = []
for i in range(len(galah_control_gaia)):
    row = galah_control_gaia.iloc[i]
    if row.sobject_id in binary_galah_ids:
        binary_idx_to_remove.append(i)
galah_control_gaia = galah_control_gaia.drop(galah_control_gaia.index[binary_idx_to_remove])
print(len(galah_control_gaia), 'remaining after removing binaries from Traven et al. 2020')

# index random set of 5000
# (just for now since it takes so long to upload to gaia)
galah_control_gaia = galah_control_gaia.sample(5000)

########### upload to gaia to filter + download RVS spectra ##################################
# query to filter based on Gaia parameters + download RVS spectra
query = f"SELECT dr3.designation, galah.sobject_id, dr3.source_id, \
galah.galah_teff, galah.galah_eteff, galah.galah_logg, galah.galah_elogg, \
galah.galah_feh, galah.galah_efeh, galah.galah_alpha, galah.galah_ealpha, \
galah.galah_vbroad, galah.galah_evbroad, dr3.rvs_spec_sig_to_noise, \
dr3.ra, dr3.dec \
FROM user_iangelo.galah_control_gaia as galah \
JOIN gaiadr3.gaia_source as dr3 \
	ON dr3.designation = galah.designation \
WHERE dr3.has_rvs = 'True' \
AND dr3.rvs_spec_sig_to_noise > 50 \
AND non_single_star = 0"

# query gaia and download RVS spectra, save to dataframes
gaia.upload_table(galah_control_gaia, 'galah_control_gaia')
galah_control_gaia_results, galah_control_flux_df, galah_control_sigma_df = gaia.retrieve_data_and_labels(query)
print('{} with has_rvs = True, snr cuts, non_single_star = 0'.format(len(galah_control_gaia_results)))

print('saving flux, flux_err to .csv files')
gaia.write_labels_to_file(galah_control_gaia_results, 'galah_control')
gaia.write_flux_data_to_csv(galah_control_flux_df, galah_control_sigma_df, 'galah_control')

