from astropy.table import Table
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# GALAH DR3 main catalog
GALAH_star_labels = Table.read('./GALAH_data_tables/GALAH_DR3_main_allstar_v2.fits', format='fits').to_pandas()
# 
GALAH_xmatch = Table.read('./GALAH_data_tables/GALAH_DR3_VAC_GaiaEDR3_v2.fits', format='fits').to_pandas()

GALAH_stars = pd.merge(GALAH_star_labels, GALAH_xmatch, on='sobject_id')

columns_to_save=['sobject_id', 'designation', 'dr3_source_id_x', 'teff', 'e_teff',
                'logg', 'e_logg', 'fe_h', 'e_fe_h', 'vbroad', 'e_vbroad', 'v_jk']
GALAH_stars[columns_to_save]

GALAH_stars[columns_to_save].rename(columns={
    "sobject_id": "galah_sobject_id", 
    "designation": "gaia_designation",
    "dr3_source_id_x":"gaia_dr3_source_id",
    "teff": "galah_teff", 
    "e_teff": "galah_eteff",
    "logg":"galah_logg",
    "e_logg":"galah_elogg",
    "fe_h": "galah_feh", 
    "e_fe_h": "galah_efeh",
    "vbroad":"galah_vbroad",
    "e_vbroad": "galah_evbroad", 
    "v_jk": "galah_vjk"})