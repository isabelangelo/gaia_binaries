# TO DO : maybe I should combine this with load_spocs_data.py
# TO DO : also add CKS here.
# TO DO : change compute_binary_detection_stats to compute_binary_metrics?
# TO DO : I need to add the relevant Gaia parameters, which requires editing the Gaia query
"""
This code computes relevant binary detection metrics
for the SPOCS sample from Brewer et al 2018
"""
import custom_model
import gaia_spectrum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# SPOCS data from Brewer et al. (2018)
spocs_labels = pd.read_csv('./data/label_dataframes/spocs_labels.csv')
spocs_flux = pd.read_csv('./data/gaia_rvs_dataframes/spocs_flux.csv')
spocs_sigma = pd.read_csv('./data/gaia_rvs_dataframes/spocs_sigma.csv')

# compute metrics
spocs_keys = [
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
	# 'radial_velocity_error',
	# 'rv_nb_transits',
	'single_fit_chisq', # oddball metrics
	'single_fit_training_density', 
	'single_fit_ca_resid'
	]

spocs_data = []
print('computing metrics for SPOCS sample from Brewer et al. (2018)')
for source_id in spocs_labels.source_id:
    print(source_id)
    flux = spocs_flux[str(source_id)]
    sigma = spocs_sigma[str(source_id)]
    spec = gaia_spectrum.GaiaSpectrum(
        source_id, 
        flux, 
        sigma, 
        model_to_use = custom_model.recent_model_version)
    spec.compute_binary_detection_stats()
    spec.compute_oddball_metrics()
    row = spocs_labels[spocs_labels.source_id==source_id].iloc[0]
    spocs_values = [
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
    # row.radial_velocity_error,
    # row.rv_nb_transits,
    spec.single_fit_chisq,
    spec.single_fit_training_density,
    spec.single_fit_ca_resid
    ]
    spocs_data.append(dict(zip(spocs_keys, spocs_values)))
spocs_df = pd.DataFrame(spocs_data)

# save data to file
spocs_df_filename = './data/binary_metric_dataframes/spocs_metrics.csv'
spocs_df.to_csv(spocs_df_filename)
print('binary detection stats for SPOCS sample saved to {}'.format(
	spocs_df_filename))


