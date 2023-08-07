from custom_model import *
import matplotlib.pyplot as plt
import os
plt.rcParams['font.size']=12

# path to save model files to, 
# should be descriptive of current model to be trained
model_fileroot = 'cabroadmask_5opt_spocssingles_traven2020binaries'
model_figure_path = './data/binary_models/'+model_fileroot+'_figures/'
os.mkdir(model_figure_path)

def plot_model_comparison(source_id, flux_df, sigma_df, object_type_str):
	"""
	Plot data + best fit binary and single star models
	for an object with a given source_id
	saves to file in ./data/binary_models/[model fileroot]_figures/
	"""
	# define function to plot calcium mask
	def plot_calcium_mask(zorder_start):
		pad = 0.02
		plt.axvspan(w[ca_idx1][0]-pad, w[ca_idx1[-1]]+pad, 
			alpha=1.0, color='#E8E8E8', zorder=zorder_start, ec='w')
		plt.axvspan(w[ca_idx2][0]-pad, w[ca_idx2[-1]]+pad, 
			alpha=1.0, color='#E8E8E8', zorder=zorder_start+1, ec='w')
		plt.axvspan(w[ca_idx3][0]-pad, w[ca_idx3[-1]]+pad, 
			alpha=1.0, color='#E8E8E8', zorder=zorder_start+2, ec='w')

	# color codes for plot
	primary_color='#91A8D6'
	secondary_color='#B03838'
	single_fit_color='#DEB23C'
	binary_fit_color = '#313DF7'

	# load flux
	source_id = str(source_id)
	flux = flux_df[source_id].to_numpy()
	sigma = sigma_df[source_id].to_numpy()

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
	q_fit = np.round(teff2mass(secondary_fit_labels[0])/teff2mass(primary_fit_labels[0]), decimals=2)
	density_fit = np.round(training_set_density(single_fit_labels), decimals=2)

	# plot figure
	plt.figure(figsize=(15,10))
	plt.subplot(311);plt.xlim(w.min(), w.max());plt.ylim(0,1.25)
	plt.errorbar(w, flux, yerr=sigma, color='k', ecolor='#E8E8E8', elinewidth=4, zorder=0)
	plt.plot(w, primary_fit, '-', color=primary_color, lw=2)
	plt.plot(w, secondary_fit, '-', color=secondary_color, lw=2)
	plt.plot(w, binary_fit, '--', color=binary_fit_color, lw=2)
	plt.text(863,1.1,'Gaia DR3 {}'.format(source_id), color='k')
	plt.text(847,0.2,'model primary', color=primary_color)
	plt.text(847,0.1,'model secondary', color=secondary_color)
	plt.text(847,1.1,'model binary: $\Delta$RV={} km/s, m$_2$/m$_1$={}, '.format(
		drv_fit, q_fit), color=binary_fit_color)
	plt.ylabel('normalized\nflux')
	plt.tick_params(axis='x', direction='inout', length=15)

	plt.subplot(312);plt.xlim(w.min(), w.max());plt.ylim(0,1.2)
	plt.errorbar(w, flux, yerr=sigma, color='k', ecolor='#E8E8E8', elinewidth=4, zorder=0)
	plt.plot(w, binary_fit, color=binary_fit_color)
	plt.plot(w, single_fit, color=single_fit_color, ls='--')
	plt.text(847,0.1,'best-fit single star\n$\chi^2={}$'.format(np.round(single_fit_chi2,2)),
	     color=single_fit_color)
	plt.text(850.5,0.1,'best-fit binary\n$\chi^2={}$'.format(np.round(binary_fit_chi2,2)),
	     color=binary_fit_color)
	plt.text(853.3,0.1,'$\Delta$ $\chi^2$={},\ntraining density={}'.format(
		np.round(single_fit_chi2-binary_fit_chi2,2),
		density_fit),
	color='dimgrey')
	plt.ylabel('normalized\nflux')
	plt.tick_params(axis='x', direction='inout', length=15)

	plt.subplot(313);plt.xlim(w.min(), w.max())
	plt.plot(w, flux - single_fit, color=single_fit_color, zorder=3)
	plt.plot(w, flux - binary_fit, color=binary_fit_color, ls='--', zorder=4)
	plot_calcium_mask(zorder_start=0)
	plt.axhline(0, color='dimgrey')
	plt.ylabel('residual')
	plt.subplots_adjust(hspace=0)
	plt.tick_params(axis='x', direction='inout', length=15)
	plt.xlabel('wavelength (nm)')

	figure_path = model_figure_path + object_type_str + '_test_case.png'
	plt.savefig(figure_path, dpi=300)

def plot_binary_metric_distributions(control_flux_df, control_sigma_df, control_label_df,
	binary_flux_df, binary_sigma_df, binary_label_df):

	def compute_metrics(flux_df, sigma_df, label_df):

		# iterate over control sample source IDs
		metric_data = []
		metric_keys = ['single_chisq', 'delta_chisq', 'training_density', 'f_imp']
		for source_id in label_df.source_id.to_numpy():
			source_id = str(source_id)
			flux = flux_df[source_id].to_numpy()
			sigma = sigma_df[source_id].to_numpy()

			# fit a binary and secondary to the data
			single_fit_labels, single_fit_chisq = fit_single_star(flux, sigma)
			binary_fit_labels, binary_fit_chisq = fit_binary(flux, sigma)

			# refine relevant metrics
			delta_chisq = single_fit_chisq - binary_fit_chisq
			training_density = training_set_density(single_fit_labels)

			# I need the binary flux and single star model flux
			single_fit  = single_star_model(single_fit_labels)
			primary_fit_labels = binary_fit_labels[:6]
			secondary_fit_labels = binary_fit_labels[6:]
			primary_fit, secondary_fit, binary_fit = binary_model(
				primary_fit_labels, 
				secondary_fit_labels,
				return_components=True)
			numerator = np.sum((np.abs(single_fit - flux) - np.abs(binary_fit - flux))/sigma)
			denominator = np.sum(np.abs(single_fit - binary_fit)/sigma)
			f_imp = numerator/denominator

			# save metrics
			metric_values = [single_fit_chisq, delta_chisq, training_density, f_imp]
			metric_data.append(dict(zip(metric_keys, metric_values)))

		metric_df = pd.DataFrame(metric_data)
		return metric_df

	control_metric_df = compute_metrics(control_flux_df, control_sigma_df, control_label_df)
	binary_metric_df = compute_metrics(binary_flux_df, binary_sigma_df, binary_label_df)

	# plot figure
	plt.rcParams['font.size']=12
	binary_color = '#EE5656'

	# histograms for all metrics
	plt.figure(figsize=(22,5));plt.tight_layout()
	plt.subplot(141)
	log_single_chisq_bins = np.linspace(3.2,4,40)
	plt.hist(np.log10(control_metric_df.single_chisq.to_numpy()), bins=log_single_chisq_bins,
	    histtype='step', color='k')
	plt.hist(np.log10(binary_metric_df.single_chisq.to_numpy()), bins=log_single_chisq_bins,
	    histtype='step', color=binary_color, lw=2.5)
	# plt.text(3.6,36,'single stars', fontsize=19)
	# plt.text(3.7,34,'binaries', color=binary_color, fontsize=19)
	plt.ylabel('number of systems', fontsize=20)
	plt.xlabel(r'log ( $\chi^2_{\rm single}$ )', fontsize=20, labelpad=15)

	plt.subplot(142)
	control_n_negative = len(control_metric_df[control_metric_df.delta_chisq<0])
	binary_n_negative = len(binary_metric_df[binary_metric_df.delta_chisq<0])
	log_delta_chisq_bins = np.linspace(-3,3,40)
	plt.hist(np.log10(control_metric_df.delta_chisq.to_numpy()), bins=log_delta_chisq_bins, 
	    histtype='step', color='k', label='N<0={}'.format(control_n_negative))
	plt.hist(np.log10(binary_metric_df.delta_chisq.to_numpy()), bins=log_delta_chisq_bins, 
	    histtype='step', color=binary_color, lw=2.5,label='N<0={}'.format(binary_n_negative))
	plt.xlabel(r'log ( $\chi^2_{\rm single}$ - $\chi^2_{\rm binary}$ )', fontsize=20, labelpad=15)
	plt.legend(frameon=False)

	plt.subplot(143)
	training_density_bins=np.linspace(0,10, 40)
	plt.hist(control_metric_df.training_density.to_numpy(), bins=training_density_bins,
	    histtype='step', color='k')
	plt.hist(binary_metric_df.training_density.to_numpy(), bins=training_density_bins,
	    histtype='step', color=binary_color, lw=2.5)
	plt.xlabel('training set\nneighbor density', fontsize=20, labelpad=10)

	plt.subplot(144)
	f_imp_bins=np.linspace(-0.4,1.2,40)
	plt.hist(control_metric_df.f_imp.to_numpy(), bins=f_imp_bins,
	    histtype='step', color='k')
	plt.hist(binary_metric_df.f_imp.to_numpy(), bins=f_imp_bins,
	    histtype='step', color=binary_color, lw=2.5)
	plt.xlabel(r'$f_{\rm imp}$', fontsize=20, labelpad=10)

	figure_path = model_figure_path + 'metric_distributions.png'
	plt.savefig(figure_path, dpi=300, bbox_inches="tight")


	# 2D distributions for different metrics
	plt.figure(figsize=(15,5));plt.tight_layout()

	plt.subplot(121)
	plt.scatter(
		binary_metric_df.single_chisq.to_numpy(), 
		binary_metric_df.delta_chisq.to_numpy()/binary_metric_df.single_chisq.to_numpy(),
		c = binary_metric_df.training_density.to_numpy(),
		vmin=0, vmax=6)
	plt.xscale('log');plt.yscale('log')
	plt.xlabel(r'$\chi^2_{\rm single}$');plt.ylabel(r'$\Delta \chi^2 / \chi^2_{\rm single}$')
	

	plt.subplot(122)
	plt.scatter(
		binary_metric_df.single_chisq.to_numpy(), 
		binary_metric_df.single_chisq.to_numpy() - binary_metric_df.delta_chisq.to_numpy(),
		c = binary_metric_df.training_density.to_numpy(),
		vmin=0, vmax=6)
	plt.colorbar(label='training set neighbor density ')
	plt.xscale('log');plt.yscale('log')
	plt.xlabel(r'$\chi^2_{\rm single}$');plt.ylabel(r'$\chi^2_{\rm binary}$')

	figure_path = model_figure_path + '2D_binary_metrics.png'
	plt.savefig(figure_path, dpi=300)

	# Figure B2 from El-Badry 2018b
	plt.figure(figsize=(10,5));plt.tight_layout()
	plt.subplot(121)
	plt.plot(control_metric_df.f_imp, np.log10(control_metric_df.delta_chisq.to_numpy()), 
		'k.', markersize=8, zorder=0, label='control sample')
	plt.plot(binary_metric_df.f_imp, np.log10(binary_metric_df.delta_chisq.to_numpy()), 
		'r.', markersize=8, label='binaries')
	plt.legend(frameon=False, loc='lower right')
	plt.xlabel(r'$f_{\rm imp}$')
	plt.ylabel(r'log ( $\chi^2_{\rm single}$ - $\chi^2_{\rm binary}$ )')

	plt.subplot(122)
	plt.plot(control_metric_df.training_density, np.log10(control_metric_df.delta_chisq.to_numpy()), 
		'k.', markersize=8, zorder=0, label='control sample')
	plt.plot(binary_metric_df.training_density, np.log10(binary_metric_df.delta_chisq.to_numpy()), 
		'r.', markersize=8, label='binaries')
	plt.legend(frameon=False, loc='lower right')
	plt.xlabel('training neighbor density')
	plt.ylabel(r'log ( $\chi^2_{\rm single}$ - $\chi^2_{\rm binary}$ )')

	figure_path = model_figure_path + 'EB2018_figureB2.png'
	plt.savefig(figure_path, dpi=300, bbox_inches="tight")

# model comparison for individual test cases

# load flux, sigma data for various samples
galah_binary_flux_df = pd.read_csv('./data/gaia_rvs_dataframes/galah_binaries_flux.csv')
galah_binary_sigma_df = pd.read_csv('./data/gaia_rvs_dataframes/galah_binaries_sigma.csv')
galah_binary_label_df = pd.read_csv('./data/galah_label_dataframes/galah_binaries_labels.csv')

raghavan_binary_flux_df = pd.read_csv('./data/gaia_rvs_dataframes/raghavan_unresolved_binaries_flux.csv')
raghavan_binary_sigma_df = pd.read_csv('./data/gaia_rvs_dataframes/raghavan_unresolved_binaries_sigma.csv')
raghavan_binary_label_df = pd.read_csv('./data/galah_label_dataframes/raghavan_unresolved_binaries_labels.csv')

raghavan_single_flux_df = pd.read_csv('./data/gaia_rvs_dataframes/raghavan_singles_flux.csv')
raghavan_single_sigma_df = pd.read_csv('./data/gaia_rvs_dataframes/raghavan_singles_sigma.csv')
raghavan_single_label_df = pd.read_csv('./data/galah_label_dataframes/raghavan_singles_labels.csv')

test_flux_df = pd.read_csv('./data/gaia_rvs_dataframes/test_flux.csv')
test_sigma_df = pd.read_csv('./data/gaia_rvs_dataframes/test_sigma.csv')

control_flux_df = pd.read_csv('./data/gaia_rvs_dataframes/galah_dopplegangers_flux.csv')
control_sigma_df = pd.read_csv('./data/gaia_rvs_dataframes/galah_dopplegangers_sigma.csv')
control_label_df = pd.read_csv('./data/galah_label_dataframes/galah_dopplegangers_labels.csv')

spocs_flux_df = pd.read_csv('../gaia_cannon_model/data/spocs_flux.csv')
spocs_sigma_df = pd.read_csv('../gaia_cannon_model/data/spocs_sigma.csv')
spocs_label_df = pd.read_csv('../gaia_cannon_model/data/spocs_labels.csv')

# make plots (this is the part to change for different samples)

# from test set + GALAH binaries
# plot_model_comparison(153768354707949056, galah_binary_flux_df, galah_binary_sigma_df, 'binary') # normal binary
# plot_model_comparison(5367424413685522688, galah_binary_flux_df, galah_binary_sigma_df, 'active_binary') # Ca emission + absorption binary
# plot_model_comparison(3798460353505152384, test_flux_df, test_sigma_df, 'single_star') # normal single star

# from Raghavan 2010 sample
# plot_model_comparison(3626268998574790656, raghavan_binary_flux_df, raghavan_binary_sigma_df, 'binary') # normal binary
# plot_model_comparison(19316224572460416, raghavan_binary_flux_df, raghavan_binary_sigma_df, 'binary2') # normal binary
# plot_model_comparison(4093301474693288960, raghavan_single_flux_df, raghavan_single_sigma_df, 'single_star') # normal single star
# plot_model_comparison(5367424413685522688, galah_binary_flux_df, galah_binary_sigma_df, 'active_binary') # Ca emission + absorption binary


# histogram of different binary detection metrics
plot_binary_metric_distributions(
	spocs_flux_df, # 3 from above that define the single star sample
	spocs_sigma_df, 
	spocs_label_df,
	galah_binary_flux_df, # 3 from above that define the binary sample
	galah_binary_sigma_df, 
	galah_binary_label_df)


