import pandas as pd
import numpy as np

# load GALAH + Gaia data for GALAH binary catalog
binaries_gaia = pd.read_csv('./GALAH_data_tables/GALAH_binaries-result.csv')
binaries_galah = pd.read_csv('./GALAH_data_tables/GALAH_binary_catalog.csv')
binaries = pd.merge(binaries_galah, binaries_gaia, on=['sobject_id','designation'], validate='one_to_one')
print(len(binaries), 'GALAH binaries from Traven et al. 2020 found in Gaia xmatch')

# require RVS spectrum, logg>4
binaries_filt = binaries.query('galah_logg>4 & has_rvs==True')
print(len(binaries_filt), ' after requiring RVS spectrum + logg>4')

# just checks, these ones don't get cut out
# require non_single_star=1
binaries_filt_non_single = binaries_filt.query('non_single_star>0')
print('{}/{} have gaia non_single_star>0'.format(len(binaries_filt_non_single), len(binaries_filt)))
binaris_filt_highSNR = binaries_filt.query('rvs_spec_sig_to_noise>50')
print('{}/{} have gaia RVS SNR>50'.format(len(binaris_filt_highSNR), len(binaries_filt)))

# write to .csv file
df_to_save = binaries_filt
filtered_sample_filename = './GALAH_data_tables/GALAH_binaries_filtered.csv'
df_to_save.to_csv(filtered_sample_filename)
print('saving filtered GALAH binary sample of {} stars to {}'.format(len(df_to_save), filtered_sample_filename))
print('')
