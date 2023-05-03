
# to do : add print statements

from astropy.table import Table
import numpy as np
import pandas as pd
import astroquery
from astroquery.vizier import Vizier


########### load data ################################################
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
galah_allstar_catalog_cleaned = galah_allstar_catalog.query('(snr_c3_iraf > 30) & (flag_sp == 0) \
& (flag_fe_h == 0) & (flag_alpha_fe == 0)')

# reformat Gaia designation column
galah_gaia_xmatch['designation'] = [i.decode().replace('EDR3', 'DR3') for i in galah_gaia_xmatch['designation']]

########### save GALAH target catalog designations ##################
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

# filters to remove stars with label uncertainties >2*median GALAH uncertainty
def emax(colname):
	return 2*np.nanmedian(galah_stars_gaia[colname])

emax_teff = emax('galah_eteff')
emax_logg = emax('galah_elogg')
emax_feh = emax('galah_efeh')
emax_alpha = emax('galah_ealpha')
emax_vbroad = emax('galah_evbroad')
galah_stars_gaia = galah_stars_gaia.query('galah_eteff<@emax_teff & galah_elogg<@emax_logg \
            & galah_efeh<@emax_feh & galah_ealpha<@emax_alpha\
            & galah_evbroad<@emax_vbroad')

# remove known binaries from training set
# note: using binary galah IDs from original vizier file yielded identical results
binary_galah_ids = galah_binary_catalog.spectID.to_numpy()
binary_idx_to_remove = []
for i in range(len(galah_stars_gaia)):
    row = galah_stars_gaia.iloc[i]
    if row.sobject_id in binary_galah_ids:
        binary_idx_to_remove.append(i)
galah_stars_gaia = galah_stars_gaia.drop(galah_stars_gaia.index[binary_idx_to_remove])

# next I need to run the gaia query to get the spectra
# where I filter based on gaia parameters




