# to do : I think that the empirical class can be an inhereted class
# from the GaiaSpectrum, but I'm not sure how at the moment.
# or at least update the GaiaSpectrum class after you have the empirical one working.
import custom_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# function to plot calcium mask
def plot_calcium_mask(zorder_start, alpha_value=0.8):
    pad = 0.02
    plt.axvspan(custom_model.w[custom_model.ca_idx1][0]-pad, 
        custom_model.w[custom_model.ca_idx1[-1]]+pad, 
        alpha=alpha_value, color='#E8E8E8', zorder=zorder_start, ec='w')
    plt.axvspan(custom_model.w[custom_model.ca_idx2][0]-pad, 
        custom_model.w[custom_model.ca_idx2[-1]]+pad, 
        alpha=alpha_value, color='#E8E8E8', zorder=zorder_start+1, ec='w')
    plt.axvspan(custom_model.w[custom_model.ca_idx3][0]-pad, 
        custom_model.w[custom_model.ca_idx3[-1]]+pad, 
        alpha=alpha_value, color='#E8E8E8', zorder=zorder_start+2, ec='w')

# spectrum object for real data
class GaiaSpectrum(object):
    """
    Gaia spectrum object
    
    Args:
        flux (np.array): simulated flux of object
        sigma (np.array): simulated flux errors of object
        type (str): 'single' or 'binary'
    """    
    def __init__(self, source_id, flux, sigma, model_to_use = custom_model.recent_model_version):
        # store input data
        self.source_id = source_id
        self.flux = flux
        self.sigma = sigma
        self.rvs_snr = np.mean(self.flux/self.sigma)

        # masked sigma for  metric calculations
        self.sigma_ca_mask = sigma.copy()
        self.sigma_ca_mask[custom_model.ca_mask] = np.inf
        
        # best-fit single star
        self.single_fit_labels, self.single_fit_chisq = custom_model.fit_single_star(
            flux, 
            sigma,
            single_star_model = model_to_use)
        self.single_fit = model_to_use(self.single_fit_labels)

        # best-fit binary
        self.binary_fit_labels, self.binary_fit_chisq = custom_model.fit_binary(
            self.flux, 
            self.sigma,
            single_star_model = model_to_use)

        # store separate primary, secondary labels (including rv)
        self.primary_fit_labels = self.binary_fit_labels[:6]
        self.secondary_fit_labels = self.binary_fit_labels[np.array([6,7,2,3,8,9])]

        # assert that the primary teff > secondary teff
        if self.primary_fit_labels[0] < self.secondary_fit_labels[0]:
            temp_labels = self.secondary_fit_labels
            self.secondary_fit_labels = self.primary_fit_labels
            self.primary_fit_labels = temp_labels
            
        # individual binary model components
        self.primary_fit, self.secondary_fit, self.binary_fit = custom_model.binary_model(
            self.primary_fit_labels, 
            self.secondary_fit_labels[np.array([0,1,4,5])],
            return_components=True,
            single_star_model = model_to_use)
        
        # fit + binary metrics
        self.delta_chisq = self.single_fit_chisq - self.binary_fit_chisq
        # compute improvement fraction
        f_imp_numerator = np.sum((np.abs(self.single_fit - self.flux) - \
            np.abs(self.binary_fit - self.flux))/self.sigma_ca_mask)
        f_imp_denominator = np.sum(np.abs(self.single_fit - self.binary_fit)/self.sigma_ca_mask)
        self.f_imp = f_imp_numerator/f_imp_denominator


    # extra metrics for binarity, moved into a function to speed up computation
    def compute_binary_metrics(self):
        self.single_fit_training_density = custom_model.training_density(self.single_fit_labels)
        self.binary_fit_drv = np.abs(self.secondary_fit_labels[-1] - self.primary_fit_labels[0])
        m1_true = custom_model.teff2mass(self.row1.teff_gspphot)
        m2_true = custom_model.teff2mass(self.row2.teff_gspphot)
        self.binary_fit_q = m2_true/m1_true

        # compute training density of binary components
        self.primary_fit_training_density = custom_model.training_density(
            self.primary_fit_labels[:-1])
        self.secondary_fit_training_density = custom_model.training_density(
            self.secondary_fit_labels[:-1])

        # compute fractional calcium line residuals
        single_fit_ca_resid_arr = (self.flux - self.single_fit)/self.sigma
        self.single_fit_ca_resid = np.sum(single_fit_ca_resid_arr[custom_model.ca_mask]**2)

    # # metrics without calcium mask, training density minimum
    # # i.e. model will fit to oddballs, albeit with incorrect labels
    # # I don't use this for the paper do I?
    # def fit_oddball(self):
    #     # fit spectrum with no constraints on model behavior
    #     oddball_fit_labels, oddball_fit_chisq = custom_model.fit_single_star(
    #         self.flux, 
    #         self.sigma,
    #         mask_calcium=False, 
    #         training_density_minimum=False)
    #     setattr(self, 'oddball_fit_labels', oddball_fit_labels)
    #     setattr(self, 'oddball_fit_chisq', oddball_fit_chisq)
    #     setattr(self, 'oddball_fit', custom_model.single_star_model(oddball_fit_labels))
    #     setattr(self, 'oddball_fit_training_density', custom_model.training_density(oddball_fit_labels))

            
    def plot(self):
        self.compute_binary_metrics()
        # color codes for plot
        primary_color='#91A8D6'
        secondary_color='#B03838'
        single_fit_color='#DEB23C'
        binary_fit_color = '#313DF7'

        # plot figure
        plt.figure(figsize=(15,10))
        plt.rcParams['font.size']=10
        plt.subplot(311);plt.xlim(custom_model.w.min(), custom_model.w.max());plt.ylim(0,1.25)
        plt.errorbar(custom_model.w, self.flux, yerr=self.sigma, color='k', ecolor='#E8E8E8', elinewidth=4, zorder=0)
        plt.plot(custom_model.w, self.primary_fit, '-', color=primary_color, lw=2)
        plt.plot(custom_model.w, self.secondary_fit, '-', color=secondary_color, lw=2)
        plt.plot(custom_model.w, self.binary_fit, '--', color=binary_fit_color, lw=2)
        plt.text(863,1.1,'Gaia DR3 {}    S/N={}'.format(self.source_id, int(self.rvs_snr)), color='k')
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

        plt.subplot(312);plt.xlim(custom_model.w.min(), custom_model.w.max());plt.ylim(0,1.2)
        plt.errorbar(custom_model.w, self.flux, yerr=self.sigma, color='k', ecolor='#E8E8E8', elinewidth=4, zorder=0)
        plt.plot(custom_model.w, self.binary_fit, color=binary_fit_color)
        plt.plot(custom_model.w, self.single_fit, color=single_fit_color, ls='--')
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

        plt.subplot(313);plt.xlim(custom_model.w.min(), custom_model.w.max())
        plt.plot(custom_model.w, self.flux - self.single_fit, color=single_fit_color, zorder=3, lw=2)
        plt.plot(custom_model.w, self.flux - self.binary_fit, color=binary_fit_color, ls='--', zorder=4)
        ca_resid_str = r'$\Sigma$(Ca resid)$^2$={}'.format(np.round(self.single_fit_ca_resid),2)
        plt.plot([],[], label = ca_resid_str, color='w', alpha=0)
        plt.legend(loc='upper right', frameon=False)
        plot_calcium_mask(zorder_start=0)
        plt.axhline(0, color='dimgrey')
        plt.ylabel('residual')
        plt.subplots_adjust(hspace=0)
        plt.tick_params(axis='x', direction='inout', length=15)
        plt.xlabel('wavelength (nm)')
        plt.ylim(-0.1,0.1)
        plt.show()

# single stars for semi-empirical binaries
single_labels = pd.read_csv('./data/label_dataframes/elbadry_singles_labels.csv')
single_flux = pd.read_csv('./data/gaia_rvs_dataframes/elbadry_singles_flux.csv')
single_sigma = pd.read_csv('./data/gaia_rvs_dataframes/elbadry_singles_sigma.csv')

# spectrum object from semi-empirical binary
# i.e., combined flux from two single stars
class SemiEmpiricalBinarySpectrum(object):
    def __init__(self, model_to_use = custom_model.recent_model_version):

        # select random primary star   
        self.row1 = single_labels.sample().iloc[0]

        # select secondary with similar feh, alpha
        single_labels_similar_met = single_labels.query(
            'abs(mh_gspphot - @self.row1.mh_gspphot)<0.05 & \
            source_id != @self.row1.source_id')

        # if there is no similar star, we can make a q=1 binary
        if len(single_labels_similar_met)==0:
            single_labels_similar_met = single_labels.query(
            'abs(mh_gspphot - @self.row1.mh_gspphot)<0.05')

        self.row2 = single_labels_similar_met.sample().iloc[0]

        # assert teff1>teff2
        if self.row1.teff_gspphot<self.row2.teff_gspphot:
            row_temp = self.row2
            self.row2 = self.row1
            self.row1 = row_temp
        
        # simulate spectrum
        # compute rv shift
        self.rv1 = 0
        self.rv2 = np.random.uniform(-10,10)
        # compute relative fluxes
        flux1_weight, flux2_weight = custom_model.flux_weights(
            self.row1.teff_gspphot, 
            self.row2.teff_gspphot)
        flux1, sigma1 = single_flux[str(self.row1.source_id)], single_sigma[str(self.row1.source_id)]
        flux2, sigma2 = single_flux[str(self.row2.source_id)], single_sigma[str(self.row2.source_id)]
        # shift flux2 according to drv
        delta_w1 = custom_model.w * self.rv1/custom_model.speed_of_light_kms
        delta_w2 = custom_model.w * self.rv2/custom_model.speed_of_light_kms
        flux1_shifted = np.interp(custom_model.w, custom_model.w + delta_w1, flux1)
        flux2_shifted = np.interp(custom_model.w, custom_model.w + delta_w2, flux2)
        # compute flux + errors
        self.primary_flux = flux1_weight*flux1_shifted
        self.secondary_flux = flux2_weight*flux2_shifted
        self.flux = self.primary_flux + self.secondary_flux
        self.sigma = flux1_weight*sigma1 + flux2_weight*sigma2
        
        # masked sigma for  metric calculations
        self.sigma_ca_mask = self.sigma.copy()
        self.sigma_ca_mask[custom_model.ca_mask] = np.inf
        
        # best-fit single star
        self.single_fit_labels, self.single_fit_chisq = custom_model.fit_single_star(
            self.flux, 
            self.sigma,
            single_star_model = model_to_use)
        self.single_fit = model_to_use(self.single_fit_labels)
        
        # best-fit binary
        self.binary_fit_labels, self.binary_fit_chisq = custom_model.fit_binary(
            self.flux, 
            self.sigma,
            single_star_model = model_to_use)

        # store separate primary, secondary labels (including rv)
        self.primary_fit_labels = self.binary_fit_labels[:6]
        self.secondary_fit_labels = self.binary_fit_labels[np.array([6,7,2,3,8,9])]

        # assert that the primary teff > secondary teff
        if self.primary_fit_labels[0] < self.secondary_fit_labels[0]:
            temp_labels = self.secondary_fit_labels
            self.secondary_fit_labels = self.primary_fit_labels
            self.primary_fit_labels = temp_labels
            
        # individual binary model components
        self.primary_fit, self.secondary_fit, self.binary_fit = custom_model.binary_model(
            self.primary_fit_labels, 
            self.secondary_fit_labels[np.array([0,1,4,5])],
            return_components=True,
            single_star_model = model_to_use)
        
    # store relevant information to reproduce El-Badry 2018
    # Figures B1-B3
    def compute_binary_detection_stats(self):
        # best-fit binary chisq
        self.delta_chisq = self.single_fit_chisq - self.binary_fit_chisq
        # compute improvement fraction
        f_imp_numerator = np.sum((np.abs(self.single_fit - self.flux) - \
            np.abs(self.binary_fit - self.flux))/self.sigma_ca_mask)
        f_imp_denominator = np.sum(np.abs(self.single_fit - self.binary_fit)/self.sigma_ca_mask)
        self.f_imp = f_imp_numerator/f_imp_denominator
        # store mass ratio
        m1_true = custom_model.teff2mass(self.row1.teff_gspphot)
        m2_true = custom_model.teff2mass(self.row2.teff_gspphot)
        self.q_true = m2_true/m1_true

        # store recovered parameters
        m1_cannon = custom_model.teff2mass(self.primary_fit_labels[0])
        m2_cannon = custom_model.teff2mass(self.secondary_fit_labels[0])
        self.q_cannon = m2_cannon/m1_cannon

        # compute training density of primary, secondary
        self.primary_fit_training_density = custom_model.training_density(
            self.primary_fit_labels[:-1])
        self.secondary_fit_training_density = custom_model.training_density(
            self.secondary_fit_labels[:-1])
    
    def compute_optimizer_stats(self):
        # compute chi-squared of binary model with true labels
        gaia_labels = ['teff_gspphot', 'logg_gspphot', 'mh_gspphot']

        # note: Gaia doesn't report alpha, and sometimes vbroad
        # in this case, I'll use the cannon values
        # and check that we found the mininum w.r.t. the other parameters
        true_alpha = self.primary_fit_labels[3]
        true_vbroad1 = self.row1['vbroad'];true_vbroad2 = self.row2['vbroad']
        if np.isnan(true_vbroad1):
            true_vbroad1 = self.primary_fit_labels[4]
        if np.isnan(true_vbroad2):
            true_vbroad2 = self.secondary_fit_labels[2]

        true_param1 = self.row1[gaia_labels].values.tolist() + [true_alpha] + \
                        [true_vbroad1] + [self.rv1]
        true_param2 = self.row2[['teff_gspphot', 'logg_gspphot']].tolist() + \
                        [true_vbroad2] + [self.rv2]

        self.true_binary_model = custom_model.binary_model(true_param1, true_param2)
        weights = 1/np.sqrt(self.sigma_ca_mask**2+custom_model.recent_model_version.s2)
        resid = weights * (self.true_binary_model - self.flux)
        self.true_binary_model_chisq = np.sum(resid**2)

# for testing code
SemiEmpiricalBinarySpectrum().compute_binary_detection_stats()

