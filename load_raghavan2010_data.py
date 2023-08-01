# note: I sorted into unresolved/resolved based on whether binaries were detected 
# as CPM or not. In reality, there still might be some "unresolved" binaries
# that could be resolved by the Gaia RVS 1.8arcsec slit width.
# also, at the moment I don't save any of the Raghavan-reported binary parameters.

from astropy.table import Table
import numpy as np
import pandas as pd
from astroquery.vizier import Vizier
import gaia

##################### load data from Raghavan 2010 table 17 ##########################################
# I computed this manually by uploading Table 17 HIP IDs
# from raghavan2010_table17_targets.txt into Gaia
raghavan_stars_gaia = pd.read_csv('./data/literature_tables/raghavan2010_table17_gaia_IDs.csv')
# store HIP ids as column to cross-rference with other tables in Raghavan 2010
raghavan_stars_gaia['HIP'] = [int(i[4:]) for i in raghavan_stars_gaia.target_id]

# query stars with RVS spectra
query = f"SELECT r2010.target_id, r2010.HIP, dr3.designation, dr3.source_id, \
dr3.rvs_spec_sig_to_noise, dr3.ra, dr3.dec, dr3.non_single_star, dr3.ruwe \
FROM user_iangelo.raghavan_stars_gaia as r2010 \
JOIN gaiadr3.gaia_source as dr3 \
	ON dr3.source_id = r2010.source_id \
WHERE dr3.has_rvs = 'True'"

# query gaia and download RVS spectra, save to dataframes
# gaia.upload_table(raghavan_stars_gaia, 'raghavan_stars_gaia')
raghavan_stars_gaia_results, flux_df, sigma_df = gaia.retrieve_data_and_labels(query)
print('{} with has_rvs = True'.format(len(raghavan_stars_gaia_results)))

##################### split into single strs, resolved + unresolved binaries #####################
def fix_dtype(table):
	"""
	fixes errors in unicode conversion from CDS tables to astropy
	"""
	for column in table.columns:
	    if table[column].dtype=='O':
	        table[column] = table[column].str.decode('utf-8')
	return(table)

# this has all the targets, listed by HIP
raghavan_table17 = Table.read('../J_ApJS_190_1_table17.dat.fits', format='fits').to_pandas()
raghavan_table17_columns = ['HD', 'HIP', 'Comp', 'SpT2', 'Mass', '[Fe/H]']
raghavan_table17 = fix_dtype(raghavan_table17)[raghavan_table17_columns] 

# this table contains all the companions found inthe binary survey
raghavan_table18 = Table.read('../J_ApJS_190_1_table18.dat.fits').to_pandas()
raghavan_table18 = fix_dtype(raghavan_table18)

binary_HIP_ids = np.unique(raghavan_table18.HIP.to_numpy())
single_HIP_ids = np.array([i for i in raghavan_table17.HIP.to_numpy() if i not in binary_HIP_ids])
print('{} singles, {} binaries, {} total stars sorted'.format(
    len(single_HIP_ids),
    len(binary_HIP_ids), 
    len(single_HIP_ids) + len(binary_HIP_ids)))

# this table contains all binaries with CPM-detected companions
# all of these targets should be resolved
raghavan_table5 = Table.read('../J_ApJS_190_1_table5.dat.fits').to_pandas()
# require larger separation than Gaia RVS slit width=1.9arcsec, remove refuted candidate companions
raghavan_table5 = fix_dtype(raghavan_table5).query("rho>1.8 & `Type` != 'RCC'")

# sort into resolved, unresolved
resolved_binary_HIP_ids = raghavan_table5.HIP.to_numpy()
unresolved_binary_HIP_ids = np.array([i for i in binary_HIP_ids if i not in resolved_binary_HIP_ids])
print('{} unresolved binaries, {} resolved binaries, {} total binaries sorted'.format(
    len(unresolved_binary_HIP_ids),
    len(resolved_binary_HIP_ids), 
    len(unresolved_binary_HIP_ids) + len(resolved_binary_HIP_ids)))

# remove HIP ids from lists if they don't have RVS spectra
# I'm sure there's a better way to do this...
has_rvs_HIP_ids = raghavan_stars_gaia_results.hip.to_numpy()
single_HIP_ids = [i for i in single_HIP_ids if i in has_rvs_HIP_ids]
unresolved_binary_HIP_ids = [i for i in unresolved_binary_HIP_ids if i in has_rvs_HIP_ids]
resolved_binary_HIP_ids = [i for i in resolved_binary_HIP_ids if i in has_rvs_HIP_ids]

# write single star, resolved + unresolved binary labels to .csv files
single_label_df = raghavan_stars_gaia_results.set_index('hip').loc[single_HIP_ids].reset_index(inplace=False)
unresolved_binary_label_df = raghavan_stars_gaia_results.set_index('hip').loc[unresolved_binary_HIP_ids].reset_index(inplace=False)
resolved_binary_label_df = raghavan_stars_gaia_results.set_index('hip').loc[resolved_binary_HIP_ids].reset_index(inplace=False)

# print numbers of stars with RVS spectra
print('{} single stars with RVS spectra'.format(len(single_label_df)))
print('{} unresolved binaries with RVS spectra'.format(len(unresolved_binary_label_df)))
print('{} resolved binaries with RVS spectra'.format(len(resolved_binary_label_df)))
print('saving flux, flux_err to .csv files')

gaia.write_labels_to_file(single_label_df, 'raghavan_singles')
gaia.write_labels_to_file(unresolved_binary_label_df , 'raghavan_unresovled_binaries')
gaia.write_labels_to_file(resolved_binary_label_df , 'raghavan_resovled_binaries')

# get the source_ids for each subgroup
single_source_ids = single_label_df.source_id.to_numpy()
unresolved_binary_source_ids = unresolved_binary_label_df.source_id.to_numpy()
resolved_binary_source_ids = resolved_binary_label_df.source_id.to_numpy()

gaia.write_flux_data_to_csv(
	flux_df[single_source_ids], 
	sigma_df[single_source_ids], 
	'raghavan_singles')

gaia.write_flux_data_to_csv(
	flux_df[unresolved_binary_source_ids],
	 sigma_df[unresolved_binary_source_ids], 
	 'raghavan_unresolved_binaries')

gaia.write_flux_data_to_csv(
	flux_df[resolved_binary_source_ids],
	 sigma_df[resolved_binary_source_ids], 
	 'raghavan_resolved_binaries')








