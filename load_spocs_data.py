from astropy.io import fits
from astropy.table import Table
import gaia
import custom_model
import thecannon as tc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# original SPOCS sample, Brewer 2016 Table 8
brewer2016_table8 = Table.read('./data/literature_data/Brewer2016/Table_8.fits', 
                               format='fits').to_pandas()
brewer2016_table9 = Table.read('./data/literature_data/Brewer2016/Table_9.fits', 
                               format='fits').to_pandas()
brewer2016_table8['Name'] = brewer2016_table8['Name'].str.decode('utf-8').str.strip() 
brewer2016_table9['Name'] = brewer2016_table9['Name'].str.decode('utf-8').str.strip() 

spocs_stars = pd.merge(
    brewer2016_table8, 
    brewer2016_table9[['Name','[Fe/H]']], 
    on='Name')
print(len(spocs_stars), ' stars in original SPOCS sample')

# edit star names to match Gaia target IDs
# (these names had to be changed for the Gaia query to work)
spocs_stars['Name'] = spocs_stars['Name'].replace('KIC-', 'KIC ', regex=True)
spocs_stars = spocs_stars.rename(columns = {"[Fe/H]":"feh"})

# remove stars that the Cannon shouldn't fit
training_label_df = pd.read_csv('./data/label_dataframes/training_labels.csv')
vbroad_min = training_label_df.galah_vbroad.min()
spocs_stars = spocs_stars.query('logg > 4 & Vbr > @vbroad_min')
print(len(spocs_stars), ' stars are main sequence (logg>4, vsini within cannon label space)')

# Gaia crossmatch
gaia_spocs_xmatch = pd.read_csv('./data/literature_data/Brewer2016/gaia_spocs_xmatch.csv')

# find SPOCS stars in crossmatch
# note: I had to remove a lot because they didn't show up 
# when I individually queried them in Gaia
spocs_stars_gaia = pd.merge(
    spocs_stars, 
    gaia_spocs_xmatch[['target_id', 'source_id']], 
    left_on='Name', 
    right_on='target_id')
print(len(spocs_stars_gaia), ' found in Gaia-SPOCS crossmatch')

# remove duplicate source ids so each row is a unique target
spocs_stars_gaia = spocs_stars_gaia.drop_duplicates(
    subset='source_id', 
    keep='first')
print(len(spocs_stars_gaia), ' remain after removing duplicate Gaia sources')

# query to run in Gaia
query = f"SELECT spocs.source_id, spocs.target_id, spocs.Teff, spocs.logg, spocs.feh, \
spocs.Vbr, dr3.rvs_spec_sig_to_noise, dr3.ra, dr3.dec, \
ap.teff_gspphot, ap.teff_gspphot_lower, ap.teff_gspphot_upper,   \
ap.logg_gspphot, ap.logg_gspphot_lower, ap.logg_gspphot_upper,   \
ap.mh_gspphot, ap.mh_gspphot_lower, ap.mh_gspphot_upper, \
ap.teff_gspspec, ap.teff_gspspec_lower, ap.teff_gspspec_upper, \
ap.logg_gspspec, ap.logg_gspspec_lower, ap.logg_gspspec_upper, \
ap.fem_gspspec, ap.fem_gspspec_lower, ap.fem_gspspec_upper, \
ap.mh_gspspec, ap.mh_gspspec_lower, ap.mh_gspspec_upper, \
ap.alphafe_gspspec, ap.alphafe_gspspec_lower, ap.alphafe_gspspec_upper, \
dr3.vbroad, dr3.vbroad_error \
FROM user_iangelo.spocs_stars_gaia as spocs  \
JOIN gaiadr3.gaia_source as dr3    \
    ON dr3.source_id = spocs.source_id \
JOIN gaiadr3.astrophysical_parameters as ap \
    ON ap.source_id = dr3.source_id \
WHERE dr3.has_rvs = 'True'"

# upload CKS star IDs to Gaia
# gaia.upload_table(spocs_stars_gaia, 'spocs_stars_gaia')

# download data
spocs_stars_gaia_results, spocs_flux_df, spocs_sigma_df = gaia.retrieve_data_and_labels(query)
print(len(spocs_stars_gaia_results), 'stars with RVS spectra in DR3')

# save data to .csv files
gaia.write_labels_to_file(spocs_stars_gaia_results, 'spocs')
gaia.write_flux_data_to_csv(spocs_flux_df, spocs_sigma_df, 'spocs')

# function to save cannon labels 
def save_spocs_cannon_labels(label_df, flux_df, sigma_df, path_to_save_labels=None):
	"""
	save set labels from SPOCS and the Cannon
	labels inferred from the training set spectra.

	Args:
	    label_df (pd.Dataframe) : literature labels of sample to plot (n_objects x n_labels)
	    flux_df (pd.Dataframe) : flux of sample to plot (n_pixels x n_objects)
	    sigma_df (pd.Dataframe) : sigma of sample to plot (n_pixels x n_objects)
	    path_to_save_labels (str) : full path to save injected + recovered labels, if given
	"""
	validation_str = 'spocs'
	pc = 'k';markersize=1;alpha_value=0.5
	full_labels = ['teff', 'logg','feh', 'alpha', 'vbroad']
	labels_to_plot = [i for i in full_labels if i!='alpha']
	validation_keys = [validation_str+'_'+ i for i in labels_to_plot] + ['rvs_spec_sig_to_noise']
	cannon_keys = ['cannon_'+ i for i in full_labels] + ['cannon_chi_sq']

	def compute_cannon_labels(label_df, flux_df, sigma_df):

	    cannon_label_data = []
	    # iterate over each object
	    for source_id in label_df.source_id.to_numpy():
	        # store galah labels
	        row = label_df.loc[label_df.source_id==source_id]
	        validation_labels = row[validation_keys].values.flatten().tolist()
	        # retrieve data
	        flux = flux_df[source_id]
	        sigma = sigma_df[source_id]
	        ivar = 1/sigma**2

	        # fit cannon model with custom optimizer
	        cannon_labels = custom_model.fit_single_star(flux, sigma)[0].tolist()
	        
	        # convert to dictionary
	        keys = ['source_id'] + validation_keys + cannon_keys
	        values = [source_id] + validation_labels + cannon_labels
	        cannon_label_data.append(dict(zip(keys, values)))

	    cannon_label_df = pd.DataFrame(cannon_label_data)
	    return cannon_label_df

	spocs_cannon_label_df = compute_cannon_labels(label_df, flux_df, sigma_df)
	if path_to_save_labels is not None:
	    cannon_label_filename = './data/label_dataframes/'+path_to_save_labels+'_labels.csv'
	    spocs_cannon_label_df.to_csv(cannon_label_filename)
	    print('cannon label dataframe saved to {}'.format(cannon_label_filename))


# save cannon label dataframe
# rename columns to be compatible with plotting function
spocs_label_df = spocs_stars_gaia_results.rename(
    columns={
    "Teff": "spocs_teff", 
    "logg": "spocs_logg",
    "feh":"spocs_feh",
    "Vbr":"spocs_vbroad",
    })

save_spocs_cannon_labels(
    spocs_label_df, 
    spocs_flux_df, 
    spocs_sigma_df, 
    path_to_save_labels = 'spocs_cannon')
