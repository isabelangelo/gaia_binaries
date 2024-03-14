# to do : add the oddball metrics to these tables,
# re-run to save.
# I can work on the single star plots while the binary code is running.
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
single_keys = [
	'apogee_id',
	'source_id', 
	'delta_chisq', 
	'f_imp',
	'training_density1', # best-fit binary parameters
	'training_density2',
	'q_cannon',
	'rv1_cannon',
	'rv2_cannon',
	'rvs_spec_sig_to_noise', # relevant Gaia parameters
	'radial_velocity_error',
	'rv_nb_transits',
    'single_fit_chisq', # oddball metrics
    'single_fit_training_density',
    'single_fit_ca_resid',
    'eq_width_849.8nm', 
    'eq_width_854.2nm',
    'eq_width_866.2nm'
	]

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
    spec.compute_best_fit_binary()
    spec.compute_binary_detection_stats()
    spec.compute_oddball_metrics()
    row = gaia_spectrum.single_labels[
    		gaia_spectrum.single_labels.source_id==source_id].iloc[0]
    single_values = [
    row.apogee_id, 
    row.source_id, 
    spec.delta_chisq, 
    spec.f_imp,
    spec.primary_fit_training_density,
    spec.secondary_fit_training_density,
    spec.q_cannon,
    spec.primary_fit_labels[5], # cannon drv1
	spec.secondary_fit_labels[5], # cannon drv2
    row.rvs_spec_sig_to_noise,
    row.radial_velocity_error,
    row.rv_nb_transits,
    spec.single_fit_chisq,
    spec.single_fit_training_density,
    spec.single_fit_ca_resid,
    spec.ca_triplet_equivalent_widths['8498Å'],
    spec.ca_triplet_equivalent_widths['8542Å'],
    spec.ca_triplet_equivalent_widths['8662Å']
    ]
    single_data.append(dict(zip(single_keys, single_values)))
single_df = pd.DataFrame(single_data)
# save data to file
single_df_filename = './data/oddball_and_binary_metric_dataframes/elbadry2018_singles_metrics.csv'
single_df.to_csv(single_df_filename)
print('binary detection stats for single stars from El-Badry 2018 saved to {}'.format(
	single_df_filename))

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
	'rv_nb_transits',
    'single_fit_chisq', # oddball metrics
    'single_fit_training_density',
    'single_fit_ca_resid'
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
    spec.compute_best_fit_binary()
    spec.compute_binary_detection_stats()
    spec.compute_oddball_metrics()
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
    row.rv_nb_transits,
    spec.single_fit_chisq,
    spec.single_fit_training_density,
    spec.single_fit_ca_resid
    ]
    binary_data.append(dict(zip(binary_keys, binary_values)))
binary_df = pd.DataFrame(binary_data)
# save data to file
binary_df_filename = './data/oddball_and_binary_metric_dataframes/elbadry2018_tableE3_binary_metrics.csv'
binary_df.to_csv(binary_df_filename)
print('binary detection stats for binaries from El-Badry 2018 Table E3 saved to {}'.format(
	binary_df_filename))






 






