from astroquery.gaia import Gaia
import numpy as np
import pandas as pd

Gaia.login()

# get full list of objects with ADQL
query = f"SELECT dr3.designation, galah.sobject_id, dr3.source_id \
FROM user_iangelo.galah_stars_filtered as galah \
JOIN gaiadr3.gaia_source as dr3 \
    ON dr3.source_id = galah.source_id"
    
job = Gaia.launch_job_async(query)
results = job.get_results()
print(f'Table size (rows): {len(results)}')

# split data up into chunks
def chunks(lst, n):
    ""
    "Split an input list into multiple chunks of size =< n"
    ""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

dl_threshold = 5000               # DataLink server threshold. It is not possible to download products for more than 5000 sources in one single call.
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


print('saving flux + flux errors to .csv files')
# save RVS spectra to .csv file
dl_key   = 'RVS_COMBINED.xml'    # Try also with 'XP_SAMPLED_COMBINED.xml'
source_id_list = []
flux_list = []
flux_error_list = []

for datalink in datalink_all:
    for i in range(len(datalink[dl_key])):
        product = datalink[dl_key][i]
        # store the gaia desgination number
        source_id_list.append(product.get_field_by_id("source_id").value) 
        prod_tab = product.to_table()
        flux_list.append(np.array(prod_tab['flux']))
        flux_error_list.append(np.array(prod_tab['flux_error']))

flux_dict = dict(zip(source_id_list, flux_list))
flux_err_dict = dict(zip(source_id_list, flux_error_list))

flux_filename = './data_files/flux_data.csv'
flux_err_filename = './data_files/flux_err_data.csv'
pd.DataFrame(flux_dict).to_csv(flux_filename, index=False)
pd.DataFrame(flux_err_dict).to_csv(flux_err_filename, index=False)
print('flux data saved to {}'.format(flux_filename))
print('flux error data saved to {}'.format(flux_err_filename))



