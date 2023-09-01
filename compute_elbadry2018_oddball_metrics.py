import spectrum
import pandas as pd

# load data for single stars, binaries from El-Badry 2018
elbadry_singles_labels = pd.read_csv('./data/label_dataframes/elbadry_singles_labels.csv')
elbadry_singles_flux = pd.read_csv('./data/gaia_rvs_dataframes/elbadry_singles_flux.csv')
elbadry_singles_sigma = pd.read_csv('./data/gaia_rvs_dataframes/elbadry_singles_sigma.csv')

elbadry_binaries_labels = pd.read_csv('./data/label_dataframes/elbadry_tableE3_binaries_labels.csv')
elbadry_binaries_flux = pd.read_csv('./data/gaia_rvs_dataframes/elbadry_tableE3_binaries_flux.csv')
elbadry_binaries_sigma = pd.read_csv('./data/gaia_rvs_dataframes/elbadry_tableE3_binaries_sigma.csv')

# function to store relevant oddball metrics
metric_keys = ['source_id', 'single_fit_chisq', 'single_fit_training_density', \
                       'single_fit_ca_resid','oddball_fit_chisq', 'oddball_fit_training_density', \
                      'rvs_spec_sig_to_noise']

def save_metric_df(label_df, flux_df, sigma_df, filename):
	metric_data = []
	for source_id in label_df.source_id:
	    flux =flux_df[str(source_id)]
	    sigma = flux_df[str(source_id)]
	    spec = spectrum.GaiaSpectrum(source_id, flux, sigma)
	    spec.fit_oddball()
	    rvs_snr = label_df[label_df.source_id == source_id].iloc[0].rvs_spec_sig_to_noise
	    metric_values = [source_id, spec.single_fit_chisq, spec.single_fit_training_density, \
	                            spec.single_fit_ca_resid, spec.oddball_fit_chisq, \
	                             spec.oddball_fit_training_density, rvs_snr]
	    metric_data.append(dict(zip(metric_keys, metric_values)))
	    print(source_id)
	    
	metric_df = pd.DataFrame(metric_data)
	metric_df.to_csv(filename)
	print('oddball metrics saved to {}'.format(filename))

oddball_metric_path = './data/oddball_metric_dataframes/'
save_metric_df(
	elbadry_singles_labels, 
	elbadry_singles_flux, 
	elbadry_singles_sigma, 
	oddball_metric_path + 'elbadry_singles.csv')

save_metric_df(
	elbadry_binaries_labels, 
	elbadry_binaries_flux, 
	elbadry_binaries_sigma, 
	oddball_metric_path + 'elbadry_binaries.csv')