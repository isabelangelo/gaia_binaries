"""
defines functions to generate the following cannon model diagnostic plots:
	plot_training_set() - histogram of training set labels
	plot_example_spec_top_panel() - labels of example training stars
	plot_exammple_spec_bottom_panel() - data + cannon model of example training stars
	one_to_one() - GALAH versus Cannon labels for sample of stars
"""
from astropy.io import fits
import custom_model
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

plt.rcParams['font.size']=12
w = fits.open('./data/cannon_training_data/gaia_rvs_wavelength.fits')[0].data[20:-20]

def plot_training_set(training_label_df, test_label_df, figure_path):
	"""
	Plot label distributions for each label in the Cannon training + test set
	saves to file in ./data/cannon_models
	"""
	plt.figure(figsize=(9,5))
	plt.rcParams['font.size']=10
	ctr = 'orange'
	cte = 'darkolivegreen'
	plt.subplot(231);plt.xlabel(r'$T_{eff}$ (K)');plt.ylabel('number of stars')
	plt.hist(training_label_df.galah_teff, color=ctr)
	plt.hist(test_label_df.galah_teff, color=cte)
	plt.text(6100,1500,'training set', color=ctr)
	plt.text(6100,1300,'test set', color=cte)
	plt.subplot(232);plt.xlabel('logg (dex)');plt.ylabel('number of stars')
	plt.hist(training_label_df.galah_logg, color=ctr)
	plt.hist(test_label_df.galah_logg, color=cte)
	plt.subplot(233);plt.xlabel('[Fe/H] (dex)');plt.ylabel('number of stars')
	plt.hist(training_label_df.galah_feh, color=ctr)
	plt.hist(test_label_df.galah_feh, color=cte)
	plt.subplot(234);plt.xlabel(r'[$\alpha$/Fe] (dex)');plt.ylabel('number of stars')
	plt.hist(training_label_df.galah_alpha, color=ctr)
	plt.hist(test_label_df.galah_alpha, color=cte)
	plt.subplot(235);plt.xlabel(r'$v_{broad}$ (km/s)');plt.ylabel('number of stars')
	plt.hist(training_label_df.galah_vbroad, color=ctr)
	plt.hist(test_label_df.galah_vbroad, color=cte)
	plt.subplots_adjust(hspace=0.3, wspace=0.4)
	plt.savefig(figure_path, dpi=300, bbox_inches='tight')


def plot_example_spec_top_panel(training_label_df, figure_path):
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
	    plt.hist(training_label_df[label_str], histtype='step', color='grey', lw=2)
	    plt.axvline(row4500[label_str], color=c4500)
	    plt.axvline(row5800[label_str], color=c5800)
	    plt.axvline(row6200[label_str], color=c6200)
	    plt.yticks([])
    
	def plot_2d_dist(label1, label2):
	    label1_str = 'galah_{}'.format(label1)
	    label2_str = 'galah_{}'.format(label2)
	    plt.plot(training_label_df[label1_str], training_label_df[label2_str], '.', color='lightgrey')
	    plt.plot(row4500[label1_str], row4500[label2_str], mec='w', ms=7, marker='o', color=c4500)
	    plt.plot(row5800[label1_str], row5800[label2_str], mec='w', ms=7, marker='o', color=c5800)
	    plt.plot(row6200[label1_str], row6200[label2_str], mec='w', ms=7, marker='o', color=c6200)


	plt.figure(figsize=(10,10))
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
	plt.savefig(figure_path, dpi=300, bbox_inches='tight')



def plot_example_spec_bottom_panel(training_label_df, flux_df, sigma_df, model, figure_path):
	"""
	Plot data + cannon model fits for 3 examples from training set
	Examples are stars with Teff=4500, 5800, 6200 K

	note: these are not the best-fit models, they are just the models based on the 
    GALAH-reported parameters 
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
	    flux = flux_df[str(row.source_id)]
	    sigma = sigma_df[str(row.source_id)]
	    ivar = 1/sigma**2
	    result = model.test(flux, ivar)
	    fit_teff, fit_logg, fit_feh, fit_alpha, fit_vbroad = result[0][0]
	    fit = result[2][0]['model_flux']
	    
	    data_str = 'GALAH Teff={}, logg={}, feh={}, alpha={}, vbroad={}'.format(
	        r(row['galah_teff']), 
	        r(row['galah_logg']), 
	        r(row['galah_feh']), 
	        r(row['galah_alpha']), 
	        r(row['galah_vbroad']))
	    
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
	    plt.ylim(0.2,1.3);plt.xlim(w.min(), w.max())
	    plt.ylabel('normalized flux')

	plt.figure(figsize=(15,10))
	plt.subplot(311);plot_spec(row6200, c6200)
	plt.subplot(312);plot_spec(row5800, c5800)
	plt.subplot(313);plot_spec(row4500, c4500)
	plt.subplots_adjust(hspace=0)
	plt.xlabel('wavelength (nm)')
	plt.savefig(figure_path, dpi=300, bbox_inches='tight')


def plot_one_to_one(label_df, flux_df, sigma_df, figure_path, path_to_save_labels=None):
	"""
	Plot a one-to-one comparison of the training set labels from GALAH and the Cannon
    labels inferred from the training set spectra.

    Args:
    	label_df (pd.Dataframe) : training labels of sample to plot (n_objects x n_labels)
    	flux_df (pd.Dataframe) : flux of sample to plot (n_pixels x n_objects)
    	sigma_df (pd.Dataframe) : sigma of sample to plot (n_pixels x n_objects)
    	model (tc.CannonModel) : cannon model object to test
    	figure_path (str) : full path to save plot to 
    	path_to_save_labels (str) : full path to save injected + recovered labels, if given
	"""
	pc = 'k';markersize=1;alpha_value=0.5
	labels_to_plot = ['galah_teff', 'galah_logg','galah_feh', 'galah_alpha', 'galah_vbroad']

	def compute_cannon_labels(label_df, flux_df, sigma_df):

		galah_keys = labels_to_plot + ['rvs_spec_sig_to_noise']

		cannon_keys = [key.replace('galah','cannon') for key in labels_to_plot] + ['cannon_chi_sq']
		cannon_label_data = []
		# iterate over each object
		for source_id in label_df.source_id.to_numpy():
			# store galah labels
			row = label_df.loc[label_df.source_id==source_id]
			galah_labels = row[galah_keys].values.flatten().tolist()
			# retrieve data
			flux = flux_df[str(source_id)]
			sigma = sigma_df[str(source_id)]

			# fit cannon model with custom optimizer
			cannon_labels = custom_model.fit_single_star(flux, sigma)[0]

			# convert to dictionary
			keys = ['source_id'] + galah_keys + cannon_keys
			values = [source_id] + galah_labels + cannon_labels.tolist()
			cannon_label_data.append(dict(zip(keys, values)))

		cannon_label_df = pd.DataFrame(cannon_label_data)
		return cannon_label_df

	def plot_label_one_to_one(label_df, label):
		x = label_df['galah_{}'.format(label)]
		y = label_df['cannon_{}'.format(label)]
		diff = y - x
		bias = np.round(np.mean(diff), 3)
		rms = np.round(np.sqrt(np.sum(diff**2)/len(diff)), 3)
		subplot_label = 'bias, rms = {}, {}'.format(bias, rms)
		plt.plot(x, y, '.', color=pc, ms=markersize, alpha=alpha_value)
		plt.plot([], [], '.', color='w', label=subplot_label)
		plt.xlabel('GALAH {}'.format(label));plt.ylabel('Cannon {}'.format(label))
		plt.plot([x.min(), x.max()], [x.min(), x.max()], lw=0.7, color='#AA8ED9')
		plt.legend(loc='upper left', frameon=False, labelcolor='firebrick')

	def plot_label_difference(label_df, label):
	    x = label_df['galah_{}'.format(label)]
	    y = label_df['cannon_{}'.format(label)]
	    diff = y - x
	    plt.hist(diff, histtype='step', color=pc)
	    plt.xlabel(r'$\Delta {}$'.format(label))

	cannon_label_df = compute_cannon_labels(
		label_df, 
		flux_df, 
		sigma_df)

	if path_to_save_labels is not None:
		cannon_label_filename = './data/label_dataframes/'+path_to_save_labels+'.csv'
		cannon_label_df.to_csv(cannon_label_filename)
		print('cannon label dataframe saved to {}'.format(cannon_label_filename))

	gs = gridspec.GridSpec(5, 2, width_ratios=[2,1])
	plt.figure(figsize=(10,17))
	for i in range(len(labels_to_plot)):
		plt.subplot(gs[2*i])
		plot_label_one_to_one(cannon_label_df, labels_to_plot[i][6:])
		plt.subplot(gs[2*i+1])
		plot_label_difference(cannon_label_df, labels_to_plot[i][6:])
	plt.savefig(figure_path, dpi=300, bbox_inches='tight')