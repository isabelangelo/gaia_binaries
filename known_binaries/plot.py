import pandas as pd
import thecannon as tc
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# load cannon model
model = tc.CannonModel.read('../train_cannon_model/cannon_models/galah_labels_5para_highSNR_cleaned.model')

# load flux data, training + test sets, cannon model
w = fits.open('../train_cannon_model/data_files/gaia_rvs_wavelength.fits')[0].data[20:-20]

plt.rcParams['figure.dpi']=150
plt.rcParams['font.size']=10

def plot_fit(source_id, flux_data_df, flux_err_data_df):
    """
    function to plot data and best-fit cannon model + residuals
    for a given source_id
    """
    s = flux_data_df[source_id][20:-20]
    serr = flux_err_data_df[source_id][20:-20]
    ivar = 1/serr**2
    
    result = model.test(s, ivar)
    fit = result[2][0]['model_flux']
    
    plt.figure(figsize=(15,10))
    plt.subplot(211);plt.title('Gaia DR3 source ID '+source_id)
    plt.errorbar(w, s, yerr=serr, lw=3, color='grey', ecolor='lightgrey', zorder=0)
    plt.plot(w, fit, color='salmon', lw=2)
    plt.ylabel('flux')
    plt.subplot(212)
    plt.plot(w, s-fit, lw=3, color='grey')
    plt.xlabel('wavelength (nm)')
    plt.ylabel('residuals')


# function that plots the labels, given the label_df
def plot_galah_vs_cannon_labels(gs, df_to_plot, pc="k", markersize=1, alpha_value=0.5):
    
    label_df = df_to_plot.copy()
    
    plt.subplot(gs[0])
    plt.plot(label_df.galah_teff, label_df.cannon_teff, '.', color=pc, ms=markersize, alpha=alpha_value)
    plt.xlabel('GALAH teff');plt.ylabel('Cannon teff')
    plt.plot([3000,9000], [3000,9000], lw=0.7, color='#AA8ED9')
    plt.subplot(gs[1]);plt.xlabel(r'$\Delta T_{\rm eff}$')
    plt.hist(label_df.galah_teff-label_df.cannon_teff, histtype='step', color=pc)

    plt.subplot(gs[2])
    plt.plot(label_df.galah_logg, label_df.cannon_logg, '.', color=pc, ms=markersize, alpha=alpha_value)
    plt.xlabel('GALAH logg');plt.ylabel('Cannon logg')
    plt.plot([3,5], [3,5], lw=0.7, color='#AA8ED9')
    plt.subplot(gs[3]);plt.xlabel(r'$\Delta$ logg')
    plt.hist(label_df.galah_logg-label_df.cannon_logg, histtype='step', color=pc)

    plt.subplot(gs[4])
    plt.plot(label_df.galah_feh, label_df.cannon_feh, '.', color=pc, ms=markersize, alpha=alpha_value)
    plt.xlabel('GALAH feh');plt.ylabel('Cannon feh')
    plt.plot([-2,2], [-2,2], lw=0.7, color='#AA8ED9')
    plt.subplot(gs[5]);plt.xlabel(r'$\Delta$[Fe/H]')
    plt.hist(label_df.galah_feh-label_df.cannon_feh, histtype='step', color=pc)

    plt.subplot(gs[6])
    plt.plot(label_df.galah_alpha_fe, label_df.cannon_alpha, '.', color=pc, ms=markersize, alpha=alpha_value)
    plt.xlabel('GALAH alpha');plt.ylabel('Cannon alpha')
    plt.plot([-1,1], [-1,1], lw=0.7, color='#AA8ED9')
    plt.subplot(gs[7]);plt.xlabel(r'$\Delta \alpha$')
    plt.hist(label_df.galah_alpha_fe-label_df.cannon_alpha, histtype='step', color=pc)

    plt.subplot(gs[8])
    plt.plot(label_df.galah_vbroad, label_df.cannon_vbroad, '.', color=pc, ms=markersize, alpha=alpha_value)
    plt.xlabel('GALAH vbroad');plt.ylabel('Cannon vbroad')
    plt.plot([0.1,100], [0.1,100], lw=0.7, color='#AA8ED9')
    plt.subplot(gs[9]);plt.xlabel(r'$\Delta v_{\rm broad}$')
    plt.hist(label_df.galah_vbroad-label_df.cannon_vbroad, histtype='step', color=pc)
    plt.subplots_adjust(hspace=0.4)


def plot_apogee_vs_cannon_labels(label_df, pc='dimgrey', label='None'):
    markersize=4
    alpha_value=0.7

    plt.subplot(131)
    plt.plot(label_df.apogee_teff, label_df.cannon_teff, '.', color=pc, ms=markersize, alpha=alpha_value,
        label=label)
    plt.xlabel('APOGEE teff');plt.ylabel('Cannon teff')
    plt.plot([3000,9000], [3000,9000], lw=0.7, color='#AA8ED9')
    plt.xlim(3000,9000)
    plt.legend()

    plt.subplot(132)
    plt.plot(label_df.apogee_logg, label_df.cannon_logg, '.', color=pc, ms=markersize, alpha=alpha_value)
    plt.xlabel('APOGEE logg');plt.ylabel('Cannon logg')
    plt.plot([3,5], [3,5], lw=0.7, color='#AA8ED9')

    plt.subplot(133)
    plt.plot(label_df.apogee_feh, label_df.cannon_feh, '.', color=pc, ms=markersize, alpha=alpha_value)
    plt.xlabel('APOGEE feh');plt.ylabel('Cannon feh')
    plt.plot([-2,2], [-2,2], lw=0.7, color='#AA8ED9')

    plt.subplots_adjust(wspace=0.3)

# plot just the test set
# gs = gridspec.GridSpec(5, 2,width_ratios=[2,1])
# df = pd.read_csv('../process_GALAH_data/GALAH_data_tables/test_set_galah_cannon_labels.csv')
# plt.figure(figsize=(10,17))
# plot_galah_vs_cannon_labels(gs, df)
# plt.savefig('/Users/isabelangelo/Desktop/label_plot.png')

# plot the test binaries
# plt.figure(figsize=(7,10))
# gs = gridspec.GridSpec(5, 2,width_ratios=[2,1])
# df1 = pd.read_csv('../process_GALAH_data/GALAH_data_tables/test_set_galah_cannon_labels.csv')
# df2 = pd.read_csv('../process_GALAH_data/GALAH_data_tables/known_binary_galah_cannon_labels.csv')
# plot_galah_vs_cannon_labels(gs, df1, pc='k', markersize=5, alpha_value=1)
# plot_galah_vs_cannon_labels(gs, df2, pc='salmon', markersize=5, alpha_value=1)
# plt.savefig('/Users/isabelangelo/Desktop/label_plot_galah_binaries2.png')

# plot the EB2018, PW2020 binaries
plt.figure(figsize=(7,10))
gs = gridspec.GridSpec(5, 2,width_ratios=[2,1])
eb2018 = pd.read_csv('./data_files/eb2018_binary_apogee_cannon_labels.csv')
pw2020 = pd.read_csv('./data_files/pw2020_binary_apogee_cannon_labels.csv')
plt.figure(figsize=(10,4))
plot_apogee_vs_cannon_labels(eb2018, pc='dodgerblue', label='El-Badry 2018')
plot_apogee_vs_cannon_labels(pw2020, pc='darkorange', label='Price-Whelan 2020')
plt.savefig('/Users/isabelangelo/Desktop/literature_label_plot.png')


