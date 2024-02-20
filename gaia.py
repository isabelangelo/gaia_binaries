"""
defines functions to query stars
and save their labels and RVS spectra to dataframes + fits files.
"""

from astropy.table import Table
import astroquery
from astroquery.gaia import Gaia
from astropy.io import fits
import pandas as pd
import numpy as np

# log in to gaia archive
Gaia.login(user='iangelo', password='#!.Eh4yShGHebfs')

# Gaia RVS wavelength (for interpolating nan fluxes)
w_interp_to = fits.open('./data/cannon_training_data/gaia_rvs_wavelength.fits')[0].data[20:-20]

# function to upload pandas table to Gaia
def upload_table(df, name):
	tbl = Table.from_pandas(df)
	Gaia.upload_table(upload_resource=tbl, table_name=name)

# much longer function to query gaia and 
# save labels, flux + sigma
def retrieve_data_and_labels(query):
	"""
	runs gaia query for stellar parameters and RVS spectra, synthesizes query output
	Args:
		query (str): Gaia query to run
	Returns:
		results (pd.DataFrame): results table output of Gaia query
		flux_df (pd.DataFrame): dataframe of RVS flux for stars in query 
								(columns are source_IDs, data are per-pixel flux values)
		sigma_df (pd.DataFrame): dataframe of RVS sigma for stars in query 
								(columns are source_IDs, data are per-pixel flux values)
	"""
	job = Gaia.launch_job_async(query)
	results = job.get_results().to_pandas()
	print(f'Table size (rows): {len(results)} after filters based on Gaia DR3 labels')

	# split data up into chunks
	def chunks(lst, n):
	    ""
	    "Split an input list into multiple chunks of size =< n"
	    ""
	    for i in range(0, len(lst), n):
	        yield lst[i:i + n]

	dl_threshold = 5000  # DataLink server threshold. It is not possible to download products for more than 5000 sources in one single call.
	ids          = results['source_id']
	ids_chunks   = list(chunks(ids, dl_threshold))
	datalink_all = []

	print(f'* Input list contains {len(ids)} source_IDs')
	print(f'* This list is split into {len(ids_chunks)} chunks of <= {dl_threshold} elements each')

	# download RVS spectra
	retrieval_type = 'RVS'        # Options are: 'EPOCH_PHOTOMETRY', 'MCMC_GSPPHOT', 'MCMC_MSC', 'XP_SAMPLED', 'XP_CONTINUOUS', 'RVS' 
	data_structure = 'COMBINED'   # Options are: 'INDIVIDUAL', 'COMBINED', 'RAW' - but as explained above, we strongly recommend to use COMBINED for massive downloads.
	data_release   = 'Gaia DR3'   # Options are: 'Gaia DR3' (default), 'Gaia DR2'
	dl_key         = f'{retrieval_type}_{data_structure}.xml'


	ii = 0
	for chunk in ids_chunks:
	    ii = ii + 1
	    print(f'Downloading Chunk #{ii}; N_files = {len(chunk)}')
	    datalink  = Gaia.load_data(ids=chunk, data_release = data_release, 
	                               retrieval_type=retrieval_type, format = 'votable', 
	                               data_structure = data_structure)
	    datalink_all.append(datalink)


	########### save RVS spectra + corresponding source ids to files ##################################
	print('writing flux + flux errors to dataframes')
	# save RVS spectra to .csv file
	dl_key   = 'RVS_COMBINED.xml'    # Try also with 'XP_SAMPLED_COMBINED.xml'
	source_id_list = []
	flux_list = []
	sigma_list = []

	for datalink in datalink_all:
	    for i in range(len(datalink[dl_key])):
	        product = datalink[dl_key][i]
	        # store the gaia desgination number
	        source_id_list.append(product.get_field_by_id("source_id").value) 
	        prod_tab = product.to_table()

	        # load flux, sigma from table
	        # pad the sides since the ends of the RVS specta have >10% nans
	       	flux = np.array(prod_tab['flux'])[20:-20]
	        sigma = np.array(prod_tab['flux_error'])[20:-20]

	        # remove nans from flux, sigma
	        # note: this needs to happen here so that the Cannon
	        # always returns flux values for all wavelengths
	        finite_idx = ~np.isnan(flux)
	        if np.sum(finite_idx) != len(flux):
        		flux = np.interp(w_interp_to, w_interp_to[finite_idx], flux[finite_idx])
        	sigma = np.nan_to_num(sigma, nan=1)

        	# append to larger flux array
        	flux_list.append(np.array(flux))
	        sigma_list.append(np.array(sigma))


	flux_df = pd.DataFrame(dict(zip(source_id_list, flux_list)))
	sigma_df = pd.DataFrame(dict(zip(source_id_list, sigma_list)))

	return results, flux_df, sigma_df


# function to write outputs to files
def write_labels_to_file(label_df, fileroot):
	path = './data/label_dataframes/'
	filename = path + fileroot + '_labels.csv'
	label_df.to_csv(filename)
	print('{} labels saved to {}'.format(fileroot, filename))


def write_flux_data_to_csv(flux_df, sigma_df, fileroot):
	path = './data/gaia_rvs_dataframes/'
	flux_filename = path + fileroot + '_flux.csv'
	sigma_filename = path + fileroot + '_sigma.csv'
	flux_df.to_csv(flux_filename)
	sigma_df.to_csv(sigma_filename)
	print('{} flux, sigma dataframe saved to:\n{}\n{}'.format(
		fileroot,
		flux_filename, 
		sigma_filename))

def write_flux_data_to_fits(flux_df, sigma_df, fileroot):
	path = './data/cannon_training_data/'
	flux_filename = path + fileroot + '_flux.fits'
	sigma_filename = path + fileroot + '_sigma.fits' 
	flux_arr = flux_df.to_numpy().T
	sigma_arr = sigma_df.to_numpy().T
	fits.HDUList([fits.PrimaryHDU(flux_arr)]).writeto(flux_filename, overwrite=True)
	fits.HDUList([fits.PrimaryHDU(sigma_arr)]).writeto(sigma_filename, overwrite=True)
	print('{} flux array saved to {}'.format(fileroot, flux_filename))
	print('{} sigma array saved to {}'.format(fileroot, sigma_filename))

