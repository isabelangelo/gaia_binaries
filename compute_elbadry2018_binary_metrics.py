"""
This code computes relevant binary detection methods
for single star and binary samples generated in load_elbadry2018_data.py
"""
import gaia_spectrum
import pandas as pd

# load data
elbadry_binary_flux_df = pd.read_csv('./data/gaia_rvs_dataframes/elbadry_tableE3_binaries_flux.csv')
elbadry_binary_sigma_df = pd.read_csv('./data/gaia_rvs_dataframes/elbadry_tableE3_binaries_sigma.csv')
elbadry_binary_label_df = pd.read_csv('./data/label_dataframes/elbadry_tableE3_binaries_labels.csv')

elbadry_single_flux_df = pd.read_csv('./data/gaia_rvs_dataframes/elbadry_singles_flux.csv')
elbadry_single_sigma_df = pd.read_csv('./data/gaia_rvs_dataframes/elbadry_singles_sigma.csv')
elbadry_single_label_df = pd.read_csv('./data/label_dataframes/elbadry_singles_labels.csv')

keys = ['source_id','single_chisq', 'binary_chisq','delta_chisq', 'training_density']
eb2018_path = './data/binary_metric_dataframes/'

# store metrics in .csv files
keys = ['source_id','single_chisq', 'binary_chisq','delta_chisq', \
        'training_density', 'binary_fit_q', 'f_imp', 'binary_fit_drv']

single_metric_data = []
for source_id in elbadry_single_label_df.source_id:
	print(source_id)
	row = elbadry_single_label_df[elbadry_single_label_df.source_id==source_id]
	snr = row.rvs_spec_sig_to_noise.iloc[0]
	flux = elbadry_single_flux_df[str(source_id)]
	sigma = elbadry_single_sigma_df[str(source_id)]
	spec = gaia_spectrum.GaiaSpectrum(source_id, flux, sigma)
	metrics = [source_id, spec.single_fit_chisq, spec.binary_fit_chisq, \
	           spec.delta_chisq, spec.single_fit_training_density, \
	           spec.binary_fit_q, spec.f_imp, spec.binary_fit_drv]
	single_metric_data.append(dict(zip(keys, metrics)))   
single_metric_df = pd.DataFrame(single_metric_data)
single_path = eb2018_path + 'single_metrics.csv'
single_metric_df.to_csv(single_path)
print('single star metrics saved to {}'.format(single_path))

binary_metric_data = []
for source_id in elbadry_binary_label_df.source_id:
	print(source_id)
	row = elbadry_binary_label_df[elbadry_binary_label_df.source_id==source_id]
	snr = row.rvs_spec_sig_to_noise.iloc[0]
	flux = elbadry_binary_flux_df[str(source_id)]
	sigma = elbadry_binary_sigma_df[str(source_id)]
	spec = gaia_spectrum.GaiaSpectrum(source_id, flux, sigma)
	metrics = [source_id, spec.single_fit_chisq, spec.binary_fit_chisq, \
	           spec.delta_chisq, spec.single_fit_training_density, \
	           spec.binary_fit_q, spec.f_imp, spec.binary_fit_drv]
	binary_metric_data.append(dict(zip(keys, metrics)))   
binary_metric_df = pd.DataFrame(binary_metric_data)
binary_path = eb2018_path + 'binary_metrics.csv'
binary_metric_df.to_csv(binary_path)
print('binary metrics saved to {}'.format(binary_path))