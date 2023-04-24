from astropy.table import Table
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astroquery
from astroquery.vizier import Vizier

########### load data ################################################
# GALAH DR3 main catalog
GALAH_star_labels = Table.read('./GALAH_data_tables/GALAH_DR3_main_allstar_v2.fits', format='fits').to_pandas()
# GALAH Gaia crossmatch for all spectra 
GALAH_xmatch = Table.read('./GALAH_data_tables/GALAH_DR3_VAC_GaiaEDR3_v2.fits', format='fits').to_pandas()
# GALAH binary catalog from Traven (2020)
catalogs = Vizier.find_catalogs('Traven')
catalogs = {k: v for k, v in catalogs.items() if 'J/A+A/638/A145' in k}
Vizier.ROW_LIMIT = -1
catalogs = Vizier.get_catalogs(catalogs.keys())
GALAH_binary_labels = catalogs[0].to_pandas()

# reformat Gaia designation column
GALAH_xmatch['designation'] = [i.decode().replace('EDR3', 'DR3') for i in GALAH_xmatch['designation']]

########### save GALAH target catalog designations ##################
# find GALAH stars in Gaia xmatch table
GALAH_star_label_cols = ['sobject_id', 'star_id', 'teff', 'e_teff', 'logg', 'e_logg', 'fe_h', 'e_fe_h',\
                             'alpha_fe', 'e_alpha_fe','vbroad', 'e_vbroad','v_jk']
GALAH_xmatch_cols = ['sobject_id', 'designation']

# note: merge on sobject_id based on GALAH website
# designations matched on multiple sobject IDs are multiple observations of the same object
# keep observation (sobj_id) per designation (this is the default in drop_duplicates())
GALAH_stars = pd.merge(GALAH_xmatch[GALAH_xmatch_cols], GALAH_star_labels[GALAH_star_label_cols], on='sobject_id')
GALAH_stars = GALAH_stars.drop_duplicates(subset='designation', keep='first')
GALAH_stars = GALAH_stars[GALAH_stars.designation!=' '] 

# save relevant parameters, write to file
GALAH_stars = GALAH_stars.rename(
    columns={
    "teff": "galah_teff", 
    "e_teff": "galah_eteff",
    "logg":"galah_logg",
    "e_logg":"galah_elogg",
    "fe_h": "galah_feh", 
    "e_fe_h": "galah_efeh",
    "alpha_fe": "galah_alpha_fe",
    "e_alpha_fe": "galah_ealpha_fe",
    "vbroad":"galah_vbroad",
    "e_vbroad": "galah_evbroad",
    "v_jk": "galah_vjk"
    })

# save to file
GALAH_stars_filename = './GALAH_data_tables/GALAH_star_catalog.csv'
GALAH_stars.to_csv(GALAH_stars_filename, index=False)
print('GALAH stars + Gaia designations saved to {}'.format(GALAH_stars_filename))


########### save GALAH binary catalog designations ###################
# merge GALAH binaries with GALAH star catalog
# to get labels for binaries when treated as single stars
GALAH_binaries_catalog = pd.merge(GALAH_binary_labels, GALAH_star_labels[GALAH_star_label_cols], 
    left_on='spectID', right_on='sobject_id', validate='one_to_one')
GALAH_binaries = pd.merge(GALAH_binaries_catalog, GALAH_xmatch[GALAH_xmatch_cols], on='sobject_id', validate='one_to_one')

# remove duplicate rows based on designation,
# keep one observation (sobject_id) per unique gaia designation
GALAH_binaries = GALAH_binaries.drop_duplicates(subset='designation', keep='first')
GALAH_binaries = GALAH_binaries[GALAH_binaries.designation!=' ']

GALAH_binaries_to_save = GALAH_binaries.rename(
    columns={
    'spectID':'sobject_id',
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
GALAH_binaries_filename = './GALAH_data_tables/GALAH_binary_catalog.csv'
GALAH_binaries_to_save.to_csv(GALAH_binaries_filename, index=False)
print('GALAH binaries + Gaia designations saved to {}'.format(GALAH_binaries_filename))

# add tests that verify that the binary xmatch is working
# number of unique sobject_id and designation should match in the final GALAH_binaries
# number of objects in final GALAH_binaries should match rows with count=1 before duplicates dropped
# len(final GALAH_binaries) + objects wtih count>1 before = len(GALAH_binaries before duplicates dropped)
# see cell in jupyter notebook scratch_workspace to see how to compute the counts










