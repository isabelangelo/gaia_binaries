"""
loads RVS spectra + labels for single stars + binaries from El-Badry et al 2018b
"""
import astropy.table as at
import pandas as pd
import numpy as np
import gaia

########### load data from El-Badry 2018 ################################################

# Table E1: targets identified as single stars
elbadry_stars = at.Table.read(
    './data/literature_data/ElBadry2018b/Table_E1_all_single_star_ids.csv')

# Table E3: targets identified as binaries in which both components contribute to the spectrum.
elbadry_binaries = at.Table.read(
    './data/literature_data/ElBadry2018b/Table_E3_all_binary_star_labels.csv')
elbadry_binaries.rename_column('APOGEE_ID', 'apogee_id')

# APOGEE DR13/Gaia DR3 crossmatch from Adrian
apogee_catalog_path = './data/literature_data/survey_catalogs/'
gaia_apogee_xmatch = at.Table.read(apogee_catalog_path + \
	'allStar-dr17-synspec-gaiadr3-gaiasourcelite.fits')
# gaia_apogee_xmatch.rename_column('APOGEE_ID', 'apogee_id')

########### crossmatch APOGEE IDs/gaia and filter sample ################################## 
# require has_rvs = True, since this information is available

# vet single stars with Gaia parameters
print('\n{} single stars listed in El-Badry 2018 Table E1'.format(len(elbadry_stars)))
elbadry_stars_gaia = at.join(elbadry_stars, gaia_apogee_xmatch, keys='apogee_id').to_pandas()
print('{} single stars found in Gaia-APOGEE crossmatch'.format(len(elbadry_stars_gaia)))
single_query_str = "non_single_star == 0 & ruwe < 1.4 & has_rvs == True"
elbadry_stars_gaia = elbadry_stars_gaia.query(single_query_str)
print('{} remain with non_single_star=0, RUWE<1.4, has_rvs=True\n'.format(len(elbadry_stars_gaia)))

# keep full binary sample
print('\n{} binaries listed in El-Badry 2018 Table E3'.format(len(elbadry_binaries)))
elbadry_binaries_gaia = at.join(elbadry_binaries, gaia_apogee_xmatch, keys='apogee_id').to_pandas()
print('{} binaries found in Gaia-APOGEE crossmatch'.format(len(elbadry_binaries_gaia)))
elbadry_binaries_gaia = elbadry_binaries_gaia.query('has_rvs == True')
print('{} remain with has_rvs=True'.format(len(elbadry_binaries_gaia)))

# remove binaries that are undetectable by our methods
# i.e., ones found in multi-epoch spectra (reported q_dyn)
elbadry_binaries_gaia = elbadry_binaries_gaia.query("(q_dyn=='---')")
print('{} binaries with no mass ratio reported from multi-epoch spectra'.format(
	len(elbadry_binaries_gaia)))


# merge full sample, preserving binary/single star labels
# store type for sorting
elbadry_stars_gaia['type'] = 'single' 
elbadry_binaries_gaia['type'] = 'binary' # store type for sorting

# make sample size the same, only preserve common columns
np.random.seed(1234) # set seed to sample single stars
single_star_sample_size = 2000 # this gets roughly 500 single stars
elbadry_stars_left = elbadry_stars_gaia.sample(n=single_star_sample_size, random_state=1234)
elbadry_binaries_right = elbadry_binaries_gaia[elbadry_stars_gaia.columns]


columns_to_keep = ['apogee_id','source_id','type']
elbadry_full_sample_gaia = pd.concat(
    (elbadry_stars_left, elbadry_binaries_right))[columns_to_keep]
print('querying sample size of N={} for single stars, binaries'.format(
	single_star_sample_size))

########### upload to gaia to download RVS spectra ##################################
query = f"SELECT eb2018.apogee_id, eb2018.source_id, dr3.designation, eb2018.type, \
dr3.rvs_spec_sig_to_noise, dr3.ra, dr3.dec, dr3.non_single_star, dr3.ruwe, \
dr3.teff_gspphot, dr3.logg_gspphot, dr3.mh_gspphot, dr3.vbroad, \
dr3.radial_velocity, dr3.radial_velocity_error, dr3.rv_nb_transits, \
dr3.phot_g_mean_mag, dr3.bp_rp \
FROM user_iangelo.elbadry_full_sample_gaia as eb2018 \
JOIN gaiadr3.gaia_source as dr3 \
ON dr3.source_id = eb2018.source_id \
WHERE dr3.has_rvs = 'True' \
AND dr3.rvs_spec_sig_to_noise > 50 \
AND dr3.logg_gspphot > 4"

# query gaia and download RVS spectra, save to dataframes
#gaia.upload_table(elbadry_full_sample_gaia, 'elbadry_full_sample_gaia')
elbadry_full_sample_gaia_results, flux_df, sigma_df = gaia.retrieve_data_and_labels(query)
print('{} out of full sample with has_rvs = True, SNR>50'.format(len(elbadry_full_sample_gaia_results)))
print('saving flux, flux_err to .csv files')

# write single + binary labels to .csv files
elbadry_stars_gaia_results = elbadry_full_sample_gaia_results.query("type == 'single' ")
elbadry_binaries_gaia_results = elbadry_full_sample_gaia_results.query("type == 'binary' ")

# merge with previous tables to store relevant quantities
elbadry_binaries_gaia_results = pd.merge(
	elbadry_binaries_gaia_results, 
	elbadry_binaries.to_pandas(), 
	on='apogee_id')

gaia.write_labels_to_file(
	elbadry_stars_gaia_results, 
	'elbadry_singles')
gaia.write_labels_to_file(
	elbadry_binaries_gaia_results, 
	'elbadry_tableE3_binaries')

# write single + binary flux, sigma to .csv files
single_source_ids = elbadry_stars_gaia_results.source_id.to_numpy()
binary_source_ids = elbadry_binaries_gaia_results.source_id.to_numpy()
print('{} are single stars, {} are binaries'.format(
	len(single_source_ids),
	len(binary_source_ids)))
	
gaia.write_flux_data_to_csv(
	flux_df[single_source_ids], 
	sigma_df[single_source_ids], 
	'elbadry_singles')

gaia.write_flux_data_to_csv(
	flux_df[binary_source_ids],
	 sigma_df[binary_source_ids], 
	 'elbadry_tableE3_binaries')


# why am I getting this error? I don't  get it iwth the de-bugger....