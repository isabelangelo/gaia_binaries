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

# model comparison for individual test cases
binary_flux_df = pd.read_csv('./data/gaia_rvs_dataframes/galah_binaries_flux.csv')
binary_sigma_df = pd.read_csv('./data/gaia_rvs_dataframes/galah_binaries_sigma.csv')
test_flux_df = pd.read_csv('./data/gaia_rvs_dataframes/test_flux.csv')
test_sigma_df = pd.read_csv('./data/gaia_rvs_dataframes/test_sigma.csv')

plot_model_comparison(153768354707949056, binary_flux_df, binary_sigma_df, 'binary') # normal binary
plot_model_comparison(5367424413685522688, binary_flux_df, binary_sigma_df, 'active_binary') # Ca emission + absorption binary
plot_model_comparison(3798460353505152384, test_flux_df, test_sigma_df, 'single_star') # normal single star



