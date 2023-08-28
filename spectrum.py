# to do : if I simulate data, 
# make this an inherited class of Spectrum
# and make another class called SimulatedSpectrum
# should I add the path as a part of the initialization
# so that it can pull the label dataframe?

from custom_model import *
import matplotlib.pyplot as plt

# function to plot calcium mask
def plot_calcium_mask(zorder_start):
    pad = 0.02
    plt.axvspan(w[ca_idx1][0]-pad, w[ca_idx1[-1]]+pad, 
        alpha=0.8, color='#E8E8E8', zorder=zorder_start, ec='w')
    plt.axvspan(w[ca_idx2][0]-pad, w[ca_idx2[-1]]+pad, 
        alpha=0.8, color='#E8E8E8', zorder=zorder_start+1, ec='w')
    plt.axvspan(w[ca_idx3][0]-pad, w[ca_idx3[-1]]+pad, 
        alpha=0.8, color='#E8E8E8', zorder=zorder_start+2, ec='w')

class GaiaSpectrum(object):
    """
    Gaia spectrum object
    
    Args:
        flux (np.array): simulated flux of object
        sigma (np.array): simulated flux errors of object
        type (str): 'single' or 'binary'
    """    
    def __init__(self, source_id, flux, sigma):
        # store input data
        self.source_id = source_id
        self.flux = flux
        self.sigma = sigma

        # masked sigma for  metric calculations
        self.sigma_ca_mask = sigma.copy()
        self.sigma_ca_mask[ca_mask] = np.inf
        
        # best-fit single star
        self.single_fit_labels, self.single_fit_chisq = fit_single_star(
            flux, 
            sigma)
        self.single_fit = single_star_model(self.single_fit_labels)
        
        # best-fit binary
        self.binary_fit_labels, self.binary_fit_chisq = fit_binary(
            flux, 
            sigma)
        self.primary_fit, self.secondary_fit, self.binary_fit = binary_model(
            self.binary_fit_labels[:6], 
            self.binary_fit_labels[6:],
            return_components=True)
        
        # fit + binary metrics
        self.delta_chisq = self.single_fit_chisq - self.binary_fit_chisq
        self.single_fit_training_density = training_density(self.single_fit_labels)
        self.binary_fit_drv = np.abs(self.binary_fit_labels[5] - self.binary_fit_labels[-1])
        binary_fit_m_arr = np.array([
            teff2mass(self.binary_fit_labels[0]),
            teff2mass(self.binary_fit_labels[6])])
        self.binary_fit_q = np.min(binary_fit_m_arr)/np.max(binary_fit_m_arr)

        # compute training density of binary components
        primary_fit_labels = self.binary_fit_labels[:5]
        secondary_fit_labels = self.binary_fit_labels[np.array([6,7,2,3,8])]
        self.primary_fit_training_density = training_density(primary_fit_labels)
        self.secondary_fit_training_density = training_density(secondary_fit_labels)

        # compute improvement fraction
        numerator = np.sum((np.abs(self.single_fit - self.flux) - \
            np.abs(self.binary_fit - self.flux))/self.sigma_ca_mask)
        denominator = np.sum(np.abs(self.single_fit - self.binary_fit)/self.sigma_ca_mask)
        self.f_imp = numerator/denominator

        # metrics without calcium mask, training density minimum
        # i.e. model will fit to oddballs, albeit with incorrect labels
    def fit_oddball(self):
        oddball_fit_labels, oddball_fit_chisq = fit_single_star(
            self.flux, 
            self.sigma,
            mask_calcium=False, 
            training_density_minimum=False)
        setattr(self, 'oddball_fit_labels', oddball_fit_labels)
        setattr(self, 'oddball_fit_chisq', oddball_fit_chisq)
        setattr(self, 'oddball_fit', single_star_model(oddball_fit_labels))
        setattr(self, 'oddball_fit_training_density', training_density(oddball_fit_labels))

            
    def plot(self):

        # color codes for plot
        primary_color='#91A8D6'
        secondary_color='#B03838'
        single_fit_color='#DEB23C'
        binary_fit_color = '#313DF7'

        # plot figure
        plt.figure(figsize=(15,10))
        plt.rcParams['font.size']=10
        plt.subplot(311);plt.xlim(w.min(), w.max());plt.ylim(0,1.25)
        plt.errorbar(w, self.flux, yerr=self.sigma, color='k', ecolor='#E8E8E8', elinewidth=4, zorder=0)
        plt.plot(w, self.primary_fit, '-', color=primary_color, lw=2)
        plt.plot(w, self.secondary_fit, '-', color=secondary_color, lw=2)
        plt.plot(w, self.binary_fit, '--', color=binary_fit_color, lw=2)
        plt.text(863,1.1,'Gaia DR3 {}'.format(self.source_id), color='k')
        plt.text(847,0.2,'model primary, Teff={}K, training density={}'.format(
            int(self.binary_fit_labels[0]),
            "{:0.2e}".format(self.primary_fit_training_density)), color=primary_color)
        plt.text(847,0.1,'model secondary, Teff={}K, training density={}'.format(
            int(self.binary_fit_labels[6]),
            "{:0.2e}".format(self.secondary_fit_training_density)), color=secondary_color)
        plt.text(847,1.1,'model binary: $\Delta$RV={} km/s, m$_2$/m$_1$={}, '.format(
            np.round(self.binary_fit_drv, decimals=2), 
            np.round(self.binary_fit_q, decimals=2)), color=binary_fit_color)
        plt.ylabel('normalized\nflux')
        plt.tick_params(axis='x', direction='inout', length=15)

        plt.subplot(312);plt.xlim(w.min(), w.max());plt.ylim(0,1.2)
        plt.errorbar(w, self.flux, yerr=self.sigma, color='k', ecolor='#E8E8E8', elinewidth=4, zorder=0)
        plt.plot(w, self.binary_fit, color=binary_fit_color)
        plt.plot(w, self.single_fit, color=single_fit_color, ls='--')
        plt.text(847,0.1,'best-fit single star\n$\chi^2={}$'.format(np.round(self.single_fit_chisq,2)),
             color=single_fit_color)
        plt.text(850.5,0.1,'best-fit binary\n$\chi^2={}$'.format(np.round(self.binary_fit_chisq,2)),
             color=binary_fit_color)
        plt.text(853.3,0.1,'$\Delta$ $\chi^2$={},\ntraining density={}'.format(
            np.round(self.delta_chisq, decimals=2),
            "{:0.2e}".format(self.single_fit_training_density)),
        color='dimgrey')
        plt.ylabel('normalized\nflux')
        plt.tick_params(axis='x', direction='inout', length=15)

        plt.subplot(313);plt.xlim(w.min(), w.max())
        plt.plot(w, self.flux - self.single_fit, color=single_fit_color, zorder=3)
        plt.plot(w, self.flux - self.binary_fit, color=binary_fit_color, ls='--', zorder=4)
        plot_calcium_mask(zorder_start=0)
        plt.axhline(0, color='dimgrey')
        plt.ylabel('residual')
        plt.subplots_adjust(hspace=0)
        plt.tick_params(axis='x', direction='inout', length=15)
        plt.xlabel('wavelength (nm)')
        plt.show()

# try to plot a star
# if it works, open a jupyter notebook to look at an oddball :)
elbadry_binary_flux_df = pd.read_csv('./data/gaia_rvs_dataframes/elbadry_tableE3_binaries_flux.csv')
elbadry_binary_sigma_df = pd.read_csv('./data/gaia_rvs_dataframes/elbadry_tableE3_binaries_sigma.csv')
elbadry_binary_label_df = pd.read_csv('./data/label_dataframes/elbadry_tableE3_binaries_labels.csv')

test_binary_source_id = 4346137540665080832
test_binary_flux = elbadry_binary_flux_df[str(test_binary_source_id)]
test_binary_sigma = elbadry_binary_sigma_df[str(test_binary_source_id)]

spec = GaiaSpectrum(test_binary_source_id, test_binary_flux, test_binary_sigma)
# import pdb;pdb.set_trace()

