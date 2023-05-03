
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
    "alpha_fe": "galah_alpha_fe",
    "e_alpha_fe": "galah_ealpha_fe",
    "vbroad":"galah_vbroad",
    "e_vbroad": "galah_evbroad",
    "v_jk": "galah_vjk"
    })

# next step: I need to now get the Gaia parameters for these
# and then filter them