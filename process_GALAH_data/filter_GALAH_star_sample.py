import pandas as pd
import numpy as np

# define list of training set labels
training_set_labels = ['galah_teff', 'galah_logg', 'galah_feh', 'galah_alpha_fe', 'galah_vbroad']

# load GALAH + Gaia data for GALAH star catalog
stars_gaia = pd.read_csv('./GALAH_data_tables/GALAH_stars-result.csv')

stars_galah = pd.read_csv('./GALAH_data_tables/GALAH_star_catalog.csv')
stars = pd.merge(stars_galah, stars_gaia, on=['sobject_id','designation'], validate='one_to_one')

# filters to remove stars with label uncertainties >2*median GALAH uncertainty
print(len(stars), 'GALAH stars from Gaia xmatch')
stars_precise_labels = stars
emax_teff = 2*np.nanmedian(stars.galah_eteff)
emax_logg = 2*np.nanmedian(stars.galah_elogg)
emax_feh = 2*np.nanmedian(stars.galah_efeh)
emax_alpha_fe = 2*np.nanmedian(stars.galah_ealpha_fe)
emax_vbroad = 2*np.nanmedian(stars.galah_evbroad)
stars_precise_labels = stars.query('galah_eteff<@emax_teff & galah_elogg<@emax_logg \
            & galah_efeh<@emax_feh & galah_ealpha_fe<@emax_alpha_fe\
            & galah_evbroad<@emax_vbroad')
print(len(stars_precise_labels), ' remaining after removing stars with label uncertainties>2xmedian GALAH uncertainty')

# filters to remove logg<4, select ones with RVS spectra, 
# and require non_single_star=0 for single star sample
stars_filt = stars_precise_labels.query('galah_logg>4 & non_single_star==0 & has_rvs==True & rvs_spec_sig_to_noise>50')
print(len(stars_filt), ' remaining after requiring logg>4, non_single_star=1, has_rvs=True, SNR>50')

# remove known binaries from training set
# note: using binary galah IDs from original vizier file yielded identical results
binary_galah_ids = pd.read_csv('./GALAH_data_tables/GALAH_binary_catalog.csv').galah_sobject_id.to_numpy()
binary_idx_to_remove = []
for i in range(len(stars_filt)):
    row = stars_filt.iloc[i]
    if row.sobject_id in binary_galah_ids:
        binary_idx_to_remove.append(i)
stars_filt_binaries_removed = stars_filt.drop(stars_filt.index[binary_idx_to_remove])
print(len(stars_filt_binaries_removed), ' remaining after removing binaries from Traven 2020')

# require training set labels to be finite
stars_finite_labels = stars_filt_binaries_removed.dropna(subset=training_set_labels)
print(len(stars_finite_labels), ' remaining after removing stars with nan training labels')

# save to .csv file
filtered_sample_filename = './GALAH_data_tables/GALAH_stars_filtered.csv'
stars_finite_labels.to_csv(filtered_sample_filename)
print('filtered GALAH star sample saved to {}'.format(filtered_sample_filename))
print('')