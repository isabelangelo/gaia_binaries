"""
This code computes relevant binary detection metrics
for single star and binary samples from Raghavan et al. 2010
generated in load_raghavan2010_data.py
"""
import custom_model
import gaia_spectrum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# binary data
binary_labels = pd.read_csv('./data/label_dataframes/raghavan_unresolved_binaries_labels.csv')
binary_flux = pd.read_csv('./data/gaia_rvs_dataframes/raghavan_unresolved_binaries_flux.csv')
binary_sigma = pd.read_csv('./data/gaia_rvs_dataframes/raghavan_unresolved_binaries_sigma.csv')
# single star data
single_labels = pd.read_csv('./data/label_dataframes/raghavan_singles_labels.csv')
single_flux = pd.read_csv('./data/gaia_rvs_dataframes/raghavan_singles_flux.csv')
single_sigma = pd.read_csv('./data/gaia_rvs_dataframes/raghavan_singles_sigma.csv')

keys = [
	'target_id',
	'source_id', 
	'delta_chisq', 
	'f_imp',
	'training_density1', # best-fit binary parameters
	'training_density2',
	'q_cannon',
	'rv1_cannon',
	'rv2_cannon',
	'rvs_spec_sig_to_noise', # relevant Gaia parameters
    'single_fit_chisq', # oddball metrics
    'single_fit_training_density',
    'single_fit_ca_resid'
	]

single_data = []
print('computing metrics for single stars from Raghavan et al. (2010)')
for source_id in single_labels.source_id:
    print(source_id)
    flux = single_flux[str(source_id)]
    sigma = single_sigma[str(source_id)]
    spec = gaia_spectrum.GaiaSpectrum(
        source_id, 
        flux, 
        sigma, 
        model_to_use = custom_model.recent_model_version)
    spec.compute_binary_detection_stats()
    spec.compute_oddball_metrics()
    row = single_labels[single_labels.source_id==source_id].iloc[0]
    values = [
    row.target_id, 
    row.source_id, 
    spec.delta_chisq, 
    spec.f_imp,
    spec.primary_fit_training_density,
    spec.secondary_fit_training_density,
    spec.q_cannon,
    spec.primary_fit_labels[5], # cannon drv1
	spec.secondary_fit_labels[5], # cannon drv2
    row.rvs_spec_sig_to_noise,
    spec.single_fit_chisq,
    spec.single_fit_training_density,
    spec.single_fit_ca_resid
    ]
    single_data.append(dict(zip(keys, values)))
single_df = pd.DataFrame(single_data)
# save data to file
single_df_filename = './data/binary_metric_dataframes/raghavan_single_metrics.csv'
single_df.to_csv(single_df_filename)
print('binary detection stats for single stars from Raghavan 2010 saved to {}'.format(
	single_df_filename))

binary_data = []
print('computing metrics for binaries from Raghavan et al. (2010)')
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
    spec.compute_oddball_metrics()
    row = binary_labels[binary_labels.source_id==source_id].iloc[0]
    values = [
    row.target_id, 
    row.source_id, 
    spec.delta_chisq, 
    spec.f_imp,
    spec.primary_fit_training_density,
    spec.secondary_fit_training_density,
    spec.q_cannon,
    spec.primary_fit_labels[5], # cannon drv1
	spec.secondary_fit_labels[5], # cannon drv2
    row.rvs_spec_sig_to_noise,
    spec.single_fit_chisq,
    spec.single_fit_training_density,
    spec.single_fit_ca_resid
    ]
    binary_data.append(dict(zip(keys, values)))
binary_df = pd.DataFrame(binary_data)
# save data to file
binary_df_filename = './data/binary_metric_dataframes/raghavan_binary_metrics.csv'
binary_df.to_csv(binary_df_filename)
print('binary detection stats for binaries from Raghavan 2010 saved to {}'.format(
	binary_df_filename))