from astropy.io import fits
from astropy.table import Table
import gaia
import custom_model
import thecannon as tc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# this has 1279, total should be 1305 so I'm not sure where the others are?
# maybe they have spectra from CKS but not derived stellar parameters.
cks_stars = pd.read_csv('./data/literature_data/cks_physical_merged.csv')
# remove duplicates + index column
cks_stars = cks_stars.drop_duplicates('id_kic').iloc[:,1:] 
print(len(cks_stars), ' unique stars in CKS sample')

# remove stars that the Cannon shouldn't fit
training_label_df = pd.read_csv('./data/label_dataframes/training_labels.csv')
vbroad_min = training_label_df.galah_vbroad.min()
cks_stars = cks_stars.query('cks_slogg > 4 & cks_svsini > @vbroad_min')
print(len(cks_stars), ' stars are main sequence (logg>4, vsini within cannon label space)')

# Gaia crossmatch
gaia_kepler_xmatch = Table.read(
    './data/literature_data/kepler_dr3_good.fits', 
    format='fits').to_pandas()

# find CKS stars in crossmatch
# note: 6 KICs from CKS are missing from the crossmatch
cks_stars_gaia = pd.merge(
    cks_stars, 
    gaia_kepler_xmatch[['kepid', 'source_id']], 
    left_on='id_kic', 
    right_on='kepid')
print(len(cks_stars_gaia), ' found in Gaia-Kepler crossmatch')

# query to run in Gaia
query = f"SELECT cks.source_id, cks.id_kic, cks.kepid, \
cks.cks_steff, cks.cks_steff_err1, cks.cks_steff_err2, \
cks.cks_slogg, cks.cks_slogg_err1, cks.cks_slogg_err2, \
cks.cks_smet, cks.cks_smet_err1, cks.cks_smet_err2, \
cks.cks_svsini, cks.cks_svsini_err1, cks.cks_svsini_err2, \
dr3.rvs_spec_sig_to_noise, dr3.ra, dr3.dec, \
ap.teff_gspphot, ap.teff_gspphot_lower, ap.teff_gspphot_upper, \
ap.logg_gspphot, ap.logg_gspphot_lower, ap.logg_gspphot_upper, \
ap.mh_gspphot, ap.mh_gspphot_lower, ap.mh_gspphot_upper, \
ap.teff_gspspec, ap.teff_gspspec_lower, ap.teff_gspspec_upper, \
ap.logg_gspspec, ap.logg_gspspec_lower, ap.logg_gspspec_upper, \
ap.fem_gspspec, ap.fem_gspspec_lower, ap.fem_gspspec_upper, \
ap.mh_gspspec, ap.mh_gspspec_lower, ap.mh_gspspec_upper, \
ap.alphafe_gspspec, ap.alphafe_gspspec_lower, ap.alphafe_gspspec_upper, \
dr3.vbroad, dr3.vbroad_error \
FROM user_iangelo.cks_stars_gaia as cks \
JOIN gaiadr3.gaia_source as dr3 \
    ON dr3.source_id = cks.source_id \
JOIN gaiadr3.astrophysical_parameters as ap \
    ON ap.source_id = dr3.source_id \
WHERE dr3.has_rvs = 'True'"

# upload CKS star IDs to Gaia
gaia.upload_table(cks_stars_gaia, 'cks_stars_gaia')

# download data
cks_stars_gaia_results, cks_flux_df, cks_sigma_df = gaia.retrieve_data_and_labels(query)
print(len(cks_stars_gaia_results), 'stars with RVS spectra in DR3')

# save data to .csv files
gaia.write_labels_to_file(cks_stars_gaia_results, 'cks')
gaia.write_flux_data_to_csv(cks_flux_df, cks_sigma_df, 'cks')

# function to save cannon labels 
def save_cks_cannon_labels(label_df, flux_df, sigma_df, path_to_save_labels=None):
	"""
	save set labels from CKS and the Cannon
	labels inferred from the training set spectra.

	Args:
	    label_df (pd.Dataframe) : literature labels of sample to plot (n_objects x n_labels)
	    flux_df (pd.Dataframe) : flux of sample to plot (n_pixels x n_objects)
	    sigma_df (pd.Dataframe) : sigma of sample to plot (n_pixels x n_objects)
	    path_to_save_labels (str) : full path to save injected + recovered labels, if given
	"""
	pc = 'k';markersize=1;alpha_value=0.5
	labels_to_plot = ['teff', 'logg','feh', 'vbroad']
	cks_keys = ['cks_teff', 'cks_logg','cks_feh', 'cks_vbroad', 'rvs_spec_sig_to_noise']
	cannon_keys = ['cannon_teff', 'cannon_logg','cannon_feh', 'cannon_alpha','cannon_vbroad',
	               'cannon_chi_sq']

	def compute_cannon_labels(label_df, flux_df, sigma_df):

	    cannon_label_data = []
	    # iterate over each object
	    for source_id in label_df.source_id.to_numpy():
	        # store galah labels
	        row = label_df.loc[label_df.source_id==source_id]
	        cks_labels = row[cks_keys].values.flatten().tolist()
	        # retrieve data
	        flux = flux_df[source_id]
	        sigma = sigma_df[source_id]
	        ivar = 1/sigma**2

	        # fit cannon model with custom optimizer
	        print('fitting custom model to Gaia DR3 {}'.format(source_id))
	        cannon_labels = custom_model.fit_single_star(flux, sigma)[0].tolist()

	        # convert to dictionary
	        keys = ['source_id'] + cks_keys + cannon_keys
	        values = [source_id] + cks_labels + cannon_labels
	        cannon_label_data.append(dict(zip(keys, values)))

	    cannon_label_df = pd.DataFrame(cannon_label_data)
	    return(cannon_label_df)

	cks_cannon_label_df = compute_cannon_labels(label_df, flux_df, sigma_df)
	if path_to_save_labels is not None:
		cannon_label_filename = './data/label_dataframes/'+path_to_save_labels+'.csv'
		cks_cannon_label_df.to_csv(cannon_label_filename)
		print('cannon label dataframe saved to {}'.format(cannon_label_filename))


# save cannon label dataframe
# rename columns to be compatible with plotting function
cks_label_df = cks_stars_gaia_results.rename(
    columns={
    "cks_steff": "cks_teff", 
    "cks_slogg": "cks_logg",
    "cks_smet":"cks_feh",
    "cks_svsini":"cks_vbroad",
    })

save_cks_cannon_labels(
    cks_label_df, 
    cks_flux_df, 
    cks_sigma_df,  
    path_to_save_labels = 'cks_cannon')
