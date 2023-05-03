from astropy.table import Table
import astroquery
from astroquery.gaia import Gaia
import pandas as pd
import numpy as np

# log in to gaia archive
Gaia.login(user='iangelo', password='@Sugargirl1994')

# function to upload pandas table to Gaia
def upload_table(df, name):
	tbl = Table.from_pandas(df)
	Gaia.upload_table(upload_resource=tbl, table_name=name)

# much longer function to query gaia and 
# save labels, flux + sigma
def retrieve_data_and_labels(query):
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
	print('saving flux + flux errors to .csv files')
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
	        flux_list.append(np.array(prod_tab['flux']))
	        sigma_list.append(np.array(prod_tab['flux_error']))

	flux_df = pd.DataFrame(dict(zip(source_id_list, flux_list)))
	sigma_df = pd.DataFrame(dict(zip(source_id_list, sigma_list)))

	return results, flux_df, sigma_df

