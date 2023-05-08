# okay, this code is ready to start implementing in the train cannon script.
# but first I need to take a break
import matplotlib.pyplot as plt
rcParams['font.size']=12
w = fits.open('./data/cannon_training_data/gaia_rvs_wavelength.fits')[0].data[20:-20]

def training_set_histogram(training_label_df, test_label_df, figure_path):
	"""
	Plot label distributions for each label in the Cannon training + test set
	saves to file in ./data/cannon_models
	"""
	plt.figure(figsize=(9,5))
	plt.rcParams['font.size']=10
	ctr = 'orange'
	cte = 'darkolivegreen'
	plt.subplot(231);xlabel(r'$T_{eff}$ (K)');ylabel('number of stars')
	plt.hist(training_label_df.galah_teff, color=ctr)
	plt.hist(test_label_df.galah_teff, color=cte)
	plt.text(6100,1500,'training set', color=ctr)
	plt.text(6100,1300,'test set', color=cte)
	plt.subplot(232);xlabel('logg (dex)');ylabel('number of stars')
	plt.hist(training_label_df.galah_logg, color=ctr)
	plt.hist(test_label_df.galah_logg, color=cte)
	plt.subplot(233);xlabel('[Fe/H] (dex)');ylabel('number of stars')
	plt.hist(training_label_df.galah_feh, color=ctr)
	plt.hist(test_label_df.galah_feh, color=cte)
	plt.subplot(234);xlabel(r'[$\alpha$/Fe] (dex)');ylabel('number of stars')
	plt.hist(training_label_df.galah_alpha, color=ctr)
	plt.hist(test_label_df.galah_alpha, color=cte)
	plt.subplot(235);xlabel(r'$v_{broad}$ (km/s)');ylabel('number of stars')
	plt.hist(training_label_df.galah_vbroad, color=ctr)
	plt.hist(test_label_df.galah_vbroad, color=cte)
	plt.subplots_adjust(hspace=0.3, wspace=0.4)
	plt.savefig(figure_path, dpi=300)


def model_plot_top_panel(training_label_df, figure_path):
	"""
	Plot label distributions with 3 examples from training set over-plotted
	Examples are stars with Teff=4500, 5800, 6200 K
	"""
	row4500 = training_label_df.iloc[np.argmin(np.abs(training_label_df.galah_teff-4500))]
	row5800 = training_label_df.iloc[np.argmin(np.abs(training_label_df.galah_teff-5800))]
	row6200 = training_label_df.iloc[np.argmin(np.abs(training_label_df.galah_teff-6200))]

	c4500 = 'tomato'
	c5800 = 'goldenrod'
	c6200 = 'cornflowerblue'

	def plot_hist(label):
	    label_str = 'galah_{}'.format(label)
	    hist(training_set[label_str], histtype='step', color='grey', lw=2)
	    axvline(row4500[label_str], color=c4500)
	    axvline(row5800[label_str], color=c5800)
	    axvline(row6200[label_str], color=c6200)
	    yticks([])
    
	def plot_2d_dist(label1, label2):
	    label1_str = 'galah_{}'.format(label1)
	    label2_str = 'galah_{}'.format(label2)
	    plot(training_set[label1_str], training_set[label2_str], '.', color='lightgrey')
	    plot(row4500[label1_str], row4500[label2_str], mec='w', ms=7, marker='o', color=c4500)
	    plot(row5800[label1_str], row5800[label2_str], mec='w', ms=7, marker='o', color=c5800)
	    plot(row6200[label1_str], row6200[label2_str], mec='w', ms=7, marker='o', color=c6200)


	figure(figsize=(10,10))
	plt.subplot(5,5,1);plot_hist('teff')
	plt.subplot(5,5,6);plt.ylabel('logg');plot_2d_dist('teff', 'logg')
	plt.subplot(5,5,7);plot_hist('logg')
	plt.subplot(5,5,11);plt.ylabel('feh');plot_2d_dist('teff', 'feh')
	plt.subplot(5,5,12);plot_2d_dist('logg', 'feh')
	plt.subplot(5,5,13);plt.xlabel('feh');plot_hist('feh')
	plt.subplot(5,5,16);plt.ylabel('alpha');plot_2d_dist('teff', 'alpha')
	plt.subplot(5,5,17);plot_2d_dist('logg', 'alpha')
	plt.subplot(5,5,18);plot_2d_dist('feh', 'alpha')
	plt.subplot(5,5,19);plt.xlabel('alpha');plot_hist('alpha')
	plt.subplot(5,5,21);plt.ylabel('vbroad');plt.xlabel('teff');plot_2d_dist('teff', 'vbroad')
	plt.subplot(5,5,22);plt.xlabel('logg');plot_2d_dist('logg', 'vbroad')
	plt.subplot(5,5,23);plt.xlabel('feh');plot_2d_dist('feh', 'vbroad')
	plt.subplot(5,5,24);plt.xlabel('alpha_fe');plot_2d_dist('alpha', 'vbroad')
	plt.subplot(5,5,25);plt.xlabel('vbroad');plot_hist('vbroad')
	plt.savefig(figure_path, dpi=300)



def model_plot_bottom_panel(training_label_df, model, figure_path):
	"""
	Plot data + cannon model fits for 3 examples from training set
	Examples are stars with Teff=4500, 5800, 6200 K
	"""
	row4500 = training_label_df.iloc[np.argmin(np.abs(training_label_df.galah_teff-4500))]
	row5800 = training_label_df.iloc[np.argmin(np.abs(training_label_df.galah_teff-5800))]
	row6200 = training_label_df.iloc[np.argmin(np.abs(training_label_df.galah_teff-6200))]

	c4500 = 'tomato'
	c5800 = 'goldenrod'
	c6200 = 'cornflowerblue'
	def r(value):
    	return np.round(value, 2)

	def plot_spec(row, crow):
	    flux = flux_data[str(row.source_id)][20:-20]
	    sigma = sigma_data[str(row.source_id)][20:-20]
	    ivar = 1/sigma**2
	    result = model.test(flux, ivar)
	    fit_teff, fit_logg, fit_feh, fit_alpha, fit_vbroad = result[0][0]
	    fit = result[2][0]['model_flux']
	    
	    data_str = 'GALAH Teff={}, logg={}, feh={}, alpha={}, vbroad={}'.format(
	        r(row.galah_teff), 
	        r(row.galah_logg), 
	        r(row.galah_feh), 
	        r(row.galah_alpha), 
	        r(row.galah_vbroad))
	    
	    fit_str = 'Cannon Teff={}, logg={}, feh={}, alpha={}, vbroad={}'.format(
	        r(fit_teff), 
	        r(fit_logg), 
	        r(fit_feh), 
	        r(fit_alpha), 
	        r(fit_vbroad))
    
	    plt.errorbar(w, flux, yerr=sigma, c='dimgrey', ecolor='lightgrey', elinewidth=4, zorder=0)
	    plt.plot(w, fit, lw=2, c=crow)
	    plt.text(864, 1.2, 'DR3 ID '+str(row.source_id), color='dimgrey')
	    plt.text(846.5, 1.2, data_str, color='dimgrey')
	    plt.text(846.5, 1.12, fit_str, color=crow)
	    plt.ylim(0.2,1.3);xlim(w.min(), w.max())
	    plt.ylabel('normalized flux')

	plt.figure(figsize=(15,10))
	plt.subplot(311);plt.plot_spec(row6200, c6200)
	plt.subplot(312);plt.plot_spec(row5800, c5800)
	plt.subplot(313);plt.plot_spec(row4500, c4500)
	plt.subplots_adjust(hspace=0)
	plt.xlabel('wavelength (nm)')
	plt.savefig(figure_path, dpi=300)





