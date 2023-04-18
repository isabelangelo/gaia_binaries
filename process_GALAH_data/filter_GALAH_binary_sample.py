import pandas as pd
import numpy as np

# load GALAH binary catalog from Gaia query
binaries = pd.read_csv('./GALAH_data_tables/GALAH_binaries-result.csv')
print(len(binaries), 'GALAH binaries from Traven et al. 2020 found in Gaia xmatch')
# require RVS spectrum
binaries_rvs = binaries.query('has_rvs==True')
print(len(binaries_rvs), ' have RVS spectra')
# require non_single_star=1
binaries_rvs_non_single = binaries_rvs.query('non_single_star==1')
print(len(binaries_rvs_non_single), ' with gaia non_single_star=1')

# write to .csv file
filtered_sample_filename = './GALAH_data_tables/GALAH_binaries_filtered.csv'
binaries_rvs_non_single.to_csv(filtered_sample_filename)
print('filtered GALAH binary sample saved to {}'.format(filtered_sample_filename))
print('')

