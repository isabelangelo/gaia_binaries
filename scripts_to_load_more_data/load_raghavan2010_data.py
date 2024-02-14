"""
loads RVS spectra + labels for single stars + binaries from Raghavan et al 2010
"""
import pandas as pd
import numpy as np
import gaia
from astropy.table import Table

# function to format data from Raghavan 2010 paper tables
def fix_dtype(table):
	"""
	fixes errors in unicode conversion from CDS tables to astropy
	"""
	for column in table.columns:
	    if table[column].dtype=='O':
	        table[column] = table[column].str.decode('utf-8')
	return(table)


########### load data from Raghavan 2010 ################################################

# full sample, with companions, not including excluded targets from Tables 2&3
raghavan_table13 = Table.read('./data/literature_data/Raghavan2010/Table_13.fits',format='fits').to_pandas()
raghavan_table13 = fix_dtype(raghavan_table13)
raghavan_table13['is_single'] = [i.isspace() for i in raghavan_table13.Comp.to_numpy()]
print(len(raghavan_table13), 'total entries in Raghavan 2010')
print(len(np.unique(raghavan_table13.HD)), 'unique HD IDs')

# identify single stars as targets with no reported companions
# also remove targets with notes=N, since this often includes stars with background sources
raghavan_stars = raghavan_table13.query("is_single==True & `N` != 'N' ").drop_duplicates('HD')
print(len(raghavan_stars), 'unique HD ids with no bound companions or background sources')

# identify binaries as targets with reported companions
raghavan_binaries = raghavan_table13.query('is_single == False')
print(len(raghavan_binaries), 'companions in Table 13')
# remove visual binaries, which are probably resolved by Gaia
raghavan_binaries['is_resolved'] = [i.isupper() for i in raghavan_binaries.Comp.to_numpy()]
raghavan_binaries = raghavan_binaries.query('is_resolved == False')
raghavan_binaries = raghavan_binaries.drop_duplicates('HD')
print(len(raghavan_binaries), 'unique HD ids after removing visual (i.e. resolved) binaries ')


########### crossmatch HIP IDs/gaia and filter sample ##################################

# I computed this manually by uploading Table 17 HIP IDs
# from Table_17_targets.txt into Gaia
raghavan_gaia_xmatch = pd.read_csv('./data/literature_data/Raghavan2010/Table_17_gaia_IDs.csv')

# get Gaia source IDs for single stars
# this step removes the sun from the single star sample, but preserves the rest
raghavan_stars['target_id'] = ['HIP '+str(i) for i in raghavan_stars.HIP]
raghavan_stars_gaia = pd.merge(
    raghavan_stars, 
    raghavan_gaia_xmatch[['target_id','designation','source_id']], 
    on='target_id') 
raghavan_stars_gaia['type'] = 'single' # store type for sorting

# get Gaia source IDs for binaries
# this step removes HIP 40167, 73695 but they don't have RVS spectra anyways
raghavan_binaries['target_id'] = ['HIP '+str(i) for i in raghavan_binaries.HIP]
raghavan_binaries_gaia = pd.merge(
    raghavan_binaries, 
    raghavan_gaia_xmatch[['target_id','designation','source_id']], 
    on='target_id') 
raghavan_binaries_gaia['type'] = 'binary'

# merge full sample, preserving binary/single star labels
# need to select list of labels, otherwise the query breaks
columns_to_keep = ['HD','OName','N','Comp','f_Comp','Per',\
                   'x_Per','Asep','Lsep','HIP','target_id',\
                   'designation','source_id','type']
raghavan_full_sample_gaia = pd.concat((raghavan_stars_gaia, raghavan_binaries_gaia))[columns_to_keep]


########### upload to gaia to download RVS spectra ##################################

query = f"SELECT r2010.target_id, r2010.HIP, dr3.designation, dr3.source_id, \
r2010.type, r2010.N, r2010.HD, dr3.rvs_spec_sig_to_noise, dr3.ra, dr3.dec, \
dr3.non_single_star, dr3.ruwe \
FROM user_iangelo.raghavan_full_sample_gaia as r2010 \
JOIN gaiadr3.gaia_source as dr3 \
ON dr3.source_id = r2010.source_id \
WHERE dr3.has_rvs = 'True' "

# query gaia and download RVS spectra, save to dataframes
gaia.upload_table(raghavan_full_sample_gaia, 'raghavan_full_sample_gaia')
raghavan_full_sample_gaia_results, flux_df, sigma_df = gaia.retrieve_data_and_labels(query)
print('{} with has_rvs = True'.format(len(raghavan_full_sample_gaia_results)))
print('saving flux, flux_err to .csv files')

# write single + binary labels to .csv files
raghavan_stars_gaia_results = raghavan_full_sample_gaia_results.query("type == 'single' ")
raghavan_binaries_gaia_results = raghavan_full_sample_gaia_results.query("type == 'binary' ")
gaia.write_labels_to_file(
	raghavan_stars_gaia_results, 
	'raghavan_singles')
gaia.write_labels_to_file(
	raghavan_binaries_gaia_results, 
	'raghavan_unresolved_binaries')


# write single + binary flux, sigma to .csv files
single_source_ids = raghavan_stars_gaia_results.source_id.to_numpy()
binary_source_ids = raghavan_binaries_gaia_results.source_id.to_numpy()
print('{} are single stars, {} are binaries'.format(
	len(single_source_ids),
	len(binary_source_ids)))
	
gaia.write_flux_data_to_csv(
	flux_df[single_source_ids], 
	sigma_df[single_source_ids], 
	'raghavan_singles')

gaia.write_flux_data_to_csv(
	flux_df[binary_source_ids],
	 sigma_df[binary_source_ids], 
	 'raghavan_unresolved_binaries')








