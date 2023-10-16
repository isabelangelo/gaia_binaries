# to do : when computing metrics, make sure to store the data from the original label table in label_dataframes/
# especailly the number of vistis, RV jitter, etc.
# to do : do I want to save anything else for the single stars? 
# there isn't much from El-Badry 2018 but maybe from Gaia/binary metrics?

"""
This code computes relevant binary detection metrics
for single star and binary samples from El-Badry 2018
generated in load_elbadry2018_data.py
"""
import custom_model
import gaia_spectrum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# binary data
binary_labels = pd.read_csv('./data/label_dataframes/elbadry_tableE3_binaries_labels.csv')
binary_flux = pd.read_csv('./data/gaia_rvs_dataframes/elbadry_tableE3_binaries_flux.csv')
binary_sigma = pd.read_csv('./data/gaia_rvs_dataframes/elbadry_tableE3_binaries_sigma.csv')

# compute metrics for single stars from El-Badry 2018
single_keys = ['delta_chisq', 'f_imp']
single_data = []
print('computing metrics for single stars from El-Badry et al. (2018)')
for source_id in gaia_spectrum.single_labels.source_id:
    print(source_id)
    flux = gaia_spectrum.single_flux[str(source_id)]
    sigma = gaia_spectrum.single_sigma[str(source_id)]
    spec = gaia_spectrum.GaiaSpectrum(
        source_id, 
        flux, 
        sigma, 
        model_to_use = custom_model.recent_model_version)
    spec.compute_binary_detection_stats()
    single_values = [spec.delta_chisq, spec.f_imp]
    single_data.append(dict(zip(single_keys, single_values)))
single_df = pd.DataFrame(single_data)
# save data to file
single_df.to_csv('./data/binary_metric_dataframes/single_metrics_leastsq_logvbroad.csv')

# compute metrics for sample of known binaries
binary_keys = [
	'apogee_id',
	'source_id',
    'delta_chisq', 
    'f_imp', 
    'teff1_true', # from Table E3
    'logg1_true', # from Table E3
    'feh_true', # from Table E3
    'mg_fe_true', # from Table E3
    'vbroad1_true', # from Table E3
    'vbroad2_true', # from Table E3
    'training_density1',
    'training_density2',
    'q_true', # from Table E3
    'q_dyn_true', # from Table E3
    'gamma_true', # from Table E3
    'teff1_cannon', 
    'logg1_cannon', 
    'feh_cannon',
    'alpha_cannon',
    'vbroad1_cannon',
    'vbroad2_cannon',
    'q_cannon', 
    'rv1_cannon',
    'rv2_cannon',
    'rvs_spec_sig_to_noise', # relevant Gaia parameters
    'radial_velocity_error',
    'rv_nb_transits'
    ]

binary_data = []
print('computing metrics for binaries from El-Badry et al. (2018)')
for source_id in binary_labels.source_id:
	print(source_id)
	flux = binary_flux[str(source_id)]
	sigma = binary_sigma[str(source_id)]
	spec = gaia_spectrum.GaiaSpectrum(
	    source_id, 
	    flux, 
	    sigma, 
	    model_to_use = custom_model.recent_model_version)
	spec.compute_binary_detection_stats()
	row = binary_labels[binary_labels.source_id==source_id].iloc[0]
	binary_values = [
	row.apogee_id,
	row.source_id,
	spec.delta_chisq, 
	spec.f_imp, 
	row['T_eff [K]'], # from Table E3
	row['log g [dex]'], # from Table E3
	row['[Fe/H] [dex]'], # from Table E3
	row['[Mg/Fe] [dex]'], # from Table E3
	row['v_macro1 [km/s]'], # from Table E3
	row['v_macro2 [km/s]'], # from Table E3
	spec.primary_fit_training_density,
	spec.secondary_fit_training_density,
	row['q_spec'], # from Table E3
	row['q_dyn'], # from Table E3
	row['gamma [km/s]'], # from Table E3
	spec.primary_fit_labels[0], # cannon teff1 
	spec.primary_fit_labels[1], # cannon logg1
	spec.primary_fit_labels[2], # cannon feh1
	spec.primary_fit_labels[3], # cannon alpha1
	spec.primary_fit_labels[4], # cannon vbroad1
	spec.secondary_fit_labels[4], # cannon vbroad2
	spec.q_cannon, 
	spec.primary_fit_labels[5], # cannon drv1
	spec.secondary_fit_labels[5], # cannon drv2
	row.rvs_spec_sig_to_noise, # relevant Gaia parameters
	row.radial_velocity_error,
	row.rv_nb_transits
	]
	binary_data.append(dict(zip(binary_keys, binary_values)))
binary_df = pd.DataFrame(binary_data)
# save data to file
binary_df.to_csv('./data/binary_metric_dataframes/binary_metrics_leastsq_logvbroad.csv')






 






