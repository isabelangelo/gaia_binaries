from binary_model import *
import matplotlib.pyplot as plt
import os
plt.rcParams['font.size']=12

# path to save model files to, 
# should be descriptive of current model to be trained
model_fileroot = 'binary_model_full'
model_figure_path = './data/binary_models/'+model_fileroot+'_figures/'
# os.mkdir(model_figure_path)

def plot_model_comparison(source_id, flux_df, sigma_df, object_type_str):
	"""
	Plot data + best fit binary and single star models
	for an object with a given source_id
	saves to file in ./data/binary_models/[model fileroot]_figures/
	"""

	# color codes for plot
	primary_color='#91A8D6'
	secondary_color='#B03838'
	single_fit_color='#DEB23C'
	binary_fit_color = '#313DF7'

	# load flux
	source_id = str(source_id)
	flux = flux_df[source_id]
	sigma = sigma_df[source_id]

	# fit single star to data
	single_fit_labels, single_fit_chi2 = fit_single_star(flux, sigma)
	single_fit  = single_star_model(single_fit_labels)

	# fit binary star to data
	binary_fit_labels, binary_fit_chi2 = fit_binary(flux, sigma)
	primary_fit_labels = binary_fit_labels[:6]
	secondary_fit_labels = binary_fit_labels[6:]
	primary_fit, secondary_fit, binary_fit = binary_model(
	        primary_fit_labels, 
	        secondary_fit_labels,
	        return_components=True)

	# compute dRV, mass ratio of best-fit binary
	drv_fit = np.round(np.abs(secondary_fit_labels[-1] - primary_fit_labels[-1]), decimals=1)
	q_fit = np.round(teff2mass(primary_fit_labels[0])/teff2mass(secondary_fit_labels[0]), decimals=2)

	# plot figure
	plt.figure(figsize=(15,10))
	plt.subplot(311);plt.xlim(w.min(), w.max());plt.ylim(0,1.25)
	plt.errorbar(w, flux, yerr=sigma, color='k', ecolor='lightgrey', elinewidth=4, zorder=0)
	plt.plot(w, primary_fit, '-', color=primary_color, lw=2)
	plt.plot(w, secondary_fit, '-', color=secondary_color, lw=2)
	plt.plot(w, binary_fit, '--', color=binary_fit_color, lw=2)
	plt.text(863,1.1,'Gaia DR3 {}'.format(source_id), color='k')
	plt.text(847,0.2,'model primary', color=primary_color)
	plt.text(847,0.1,'model secondary', color=secondary_color)
	plt.text(847,1.1,'model binary: $\Delta$RV={} km/s, m$_2$/m$_1$={}'.format(
		drv_fit, q_fit), color=binary_fit_color)
	plt.ylabel('normalized\nflux')
	plt.tick_params(axis='x', direction='inout', length=15)

	plt.subplot(312);plt.xlim(w.min(), w.max());plt.ylim(0,1.2)
	plt.errorbar(w, flux, yerr=sigma, color='k', ecolor='lightgrey', elinewidth=4, zorder=0)
	plt.plot(w, binary_fit, color=binary_fit_color)
	plt.plot(w, single_fit, color=single_fit_color, ls='--')
	plt.text(847,0.1,'best-fit single star\n$\chi^2={}$'.format(np.round(single_fit_chi2,2)),
	     color=single_fit_color)
	plt.text(850.5,0.1,'best-fit binary\n$\chi^2={}$'.format(np.round(binary_fit_chi2,2)),
	     color=binary_fit_color)
	plt.text(853.3,0.2,'$\Delta\chi^2={}$'.format(np.round(single_fit_chi2-binary_fit_chi2,2)),
	     color='dimgrey')
	plt.ylabel('normalized\nflux')
	plt.tick_params(axis='x', direction='inout', length=15)

	plt.subplot(313);plt.xlim(w.min(), w.max())
	plt.plot(w, flux - single_fit, color=single_fit_color)
	plt.plot(w, flux - binary_fit, color=binary_fit_color, ls='--')
	plt.axhline(0, color='dimgrey')
	plt.ylabel('residual')
	plt.subplots_adjust(hspace=0)
	plt.tick_params(axis='x', direction='inout', length=15)
	plt.xlabel('wavelength (nm)')

	figure_path = model_figure_path + object_type_str + '_test_case.png'
	plt.savefig(figure_path, dpi=300)

def plot_binary_metric_distributions():

	# load data for control sample
	control_flux_df = pd.read_csv('./data/gaia_rvs_dataframes/galah_dopplegangers_flux.csv')
	control_sigma_df = pd.read_csv('./data/gaia_rvs_dataframes/galah_dopplegangers_sigma.csv')
	control_label_df = pd.read_csv('./data/galah_label_dataframes/galah_dopplegangers_labels.csv')
	# load data for binary sample
	binary_flux_df = pd.read_csv('./data/gaia_rvs_dataframes/galah_binaries_flux.csv')
	binary_sigma_df = pd.read_csv('./data/gaia_rvs_dataframes/galah_binaries_sigma.csv')
	binary_label_df = pd.read_csv('./data/galah_label_dataframes/galah_binaries_labels.csv')

	def compute_metrics(flux_df, sigma_df, label_df):

		# iterate over control sample source IDs
		metric_data = []
		metric_keys = ['single_chisq', 'delta_chisq', 'training_density']
		for source_id in label_df.source_id.to_numpy():
			source_id = str(source_id)
			flux = flux_df[source_id]
			sigma = sigma_df[source_id]

			# fit a binary and secondary to the data
			single_fit_labels, single_fit_chisq = fit_single_star(flux, sigma)
			binary_fit_labels, binary_fit_chisq = fit_binary(flux, sigma)

			# refine relevant metrics
			delta_chisq = single_fit_chisq - binary_fit_chisq
			training_density = training_set_density(single_fit_labels)

			# save metrics
			metric_values = [single_fit_chisq, delta_chisq, training_density]
			metric_data.append(dict(zip(metric_keys, metric_values)))

		metric_df = pd.DataFrame(metric_data)
		return metric_df

	control_metric_df = compute_metrics(control_flux_df, control_sigma_df, control_label_df)
	binary_metric_df = compute_metrics(binary_flux_df, binary_sigma_df, binary_label_df)

	# plot figure
	plt.rcParams['font.size']=15
	binary_color = '#EE5656'

	plt.figure(figsize=(17,10))
	plt.subplot(131)
	log_single_chisq_bins = np.linspace(3.2,4,40)
	plt.hist(np.log10(control_metric_df.single_chisq.to_numpy()), bins=log_single_chisq_bins,
	    histtype='step', color='k')
	plt.hist(np.log10(binary_metric_df.single_chisq.to_numpy()), bins=log_single_chisq_bins,
	    histtype='step', color=binary_color, lw=2.5)
	plt.text(3.6,38,'single stars', fontsize=19)
	plt.text(3.7,34,'binaries', color=binary_color, fontsize=19)
	plt.ylabel('number of systems', fontsize=20)
	plt.xlabel(r'log ( $\chi^2_{\rm single}$ )', fontsize=20, labelpad=15)

	plt.subplot(132)
	log_delta_chisq_bins = np.linspace(-3,3,40)
	plt.hist(np.log10(control_metric_df.delta_chisq.to_numpy()), bins=log_delta_chisq_bins, 
	    histtype='step', color='k')
	plt.hist(np.log10(binary_metric_df.delta_chisq.to_numpy()), bins=log_delta_chisq_bins, 
	    histtype='step', color=binary_color, lw=2.5)
	plt.xlabel(r'log ( $\chi^2_{\rm single}$ - $\chi^2_{\rm binary}$ )', fontsize=20, labelpad=15)

	plt.subplot(133)
	training_density_bins=np.linspace(0,10, 40)
	plt.hist(control_metric_df.training_density.to_numpy(), bins=training_density_bins,
	    histtype='step', color='k')
	plt.hist(binary_metric_df.training_density.to_numpy(), bins=training_density_bins,
	    histtype='step', color=binary_color, lw=2.5)
	plt.xlabel('training set\nneighbor density', fontsize=20, labelpad=10)

	figure_path = model_figure_path + 'metric_distributions.png'
	plt.savefig(figure_path, dpi=300)


# model comparison for individual test cases
binary_flux_df = pd.read_csv('./data/gaia_rvs_dataframes/galah_binaries_flux.csv')
binary_sigma_df = pd.read_csv('./data/gaia_rvs_dataframes/galah_binaries_sigma.csv')
test_flux_df = pd.read_csv('./data/gaia_rvs_dataframes/test_flux.csv')
test_sigma_df = pd.read_csv('./data/gaia_rvs_dataframes/test_sigma.csv')
plot_model_comparison(153768354707949056, binary_flux_df, binary_sigma_df, 'binary') # normal binary
plot_model_comparison(5367424413685522688, binary_flux_df, binary_sigma_df, 'active_binary') # Ca emission + absorption binary
plot_model_comparison(3798460353505152384, test_flux_df, test_sigma_df, 'single_star') # normal single star

# historgram of different binary detection metrics
# note: maybe at one point I'll want this function to take input samples
# so I can easily compare two different samples
plot_binary_metric_distributions()


