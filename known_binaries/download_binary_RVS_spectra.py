from astroquery.gaia import Gaia
import numpy as np
import pandas as pd

Gaia.login()

# get full list of objects with ADQL
query = f"SELECT dr3.designation, galah.sobject_id, dr3.source_id \
FROM user_iangelo.galah_binaries_filtered as galah \
JOIN gaiadr3.gaia_source as dr3 \
    ON dr3.source_id = galah.source_id"

pw2020_query = f"SELECT dr3.designation, pw2020.apogee_id, dr3.source_id \
FROM user_iangelo.pricewhelan2020_binaries_filtered as pw2020 \
JOIN gaiadr3.gaia_source as dr3 \
    ON dr3.source_id = pw2020.gaiadr3_source_id \
WHERE dr3.rvs_spec_sig_to_noise > 50"

job = Gaia.launch_job_async(pw2020_query)
results = job.get_results()
print(f'Table size (rows): {len(results)}')

# question: how do I do this without chunks?

# download RVS spectra
retrieval_type = 'RVS'        # Options are: 'EPOCH_PHOTOMETRY', 'MCMC_GSPPHOT', 'MCMC_MSC', 'XP_SAMPLED', 'XP_CONTINUOUS', 'RVS' 
data_structure = 'COMBINED'   # Options are: 'INDIVIDUAL', 'COMBINED', 'RAW' - but as explained above, we strongly recommend to use COMBINED for massive downloads.
data_release   = 'Gaia DR3'   # Options are: 'Gaia DR3' (default), 'Gaia DR2'


datalink  = Gaia.load_data(ids=results['source_id'], data_release = data_release, 
                               retrieval_type=retrieval_type, format = 'votable', 
                               data_structure = data_structure, output_file = None)
dl_keys  = [inp for inp in datalink.keys()]
dl_keys.sort()

print()
print(f'The following Datalink products have been downloaded:')
for dl_key in dl_keys:
    print(f' * {dl_key}')


print('saving flux + flux errors to .csv files')
# save RVS spectra to .csv file
dl_key   = 'RVS_COMBINED.xml'    # Try also with 'XP_SAMPLED_COMBINED.xml'
source_id_list = []
flux_list = []
flux_error_list = []

for i in range(len(datalink[dl_key])):
    product = datalink[dl_key][i]
    # store the gaia desgination number
    source_id_list.append(product.get_field_by_id("source_id").value) 
    prod_tab = product.to_table()
    flux_list.append(np.array(prod_tab['flux']))
    flux_error_list.append(np.array(prod_tab['flux_error']))

flux_dict = dict(zip(source_id_list, flux_list))
flux_err_dict = dict(zip(source_id_list, flux_error_list))

flux_filename = './data_files/flux_data_pw2020.csv' # flux_data_binaries
flux_err_filename = './data_files/flux_err_data_pw2020.csv' # flux_err_data_binaries
pd.DataFrame(flux_dict).to_csv(flux_filename, index=False)
pd.DataFrame(flux_err_dict).to_csv(flux_err_filename, index=False)
print('flux data saved to {}'.format(flux_filename))
print('flux error data saved to {}'.format(flux_err_filename))




