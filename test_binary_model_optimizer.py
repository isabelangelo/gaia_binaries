import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import custom_model

np.random.seed(1234)

test_labels = pd.read_csv('./data/label_dataframes/test_labels.csv')
test_flux = pd.read_csv('./data/gaia_rvs_dataframes/test_flux.csv')
test_sigma = pd.read_csv('./data/gaia_rvs_dataframes/test_sigma.csv')

class SemiEmpiricalBinarySpectrum(object):
    def __init__(self):

    	# select random primary star   
        self.row1 = test_labels.sample().iloc[0]

    	# select secondary with similar feh, alpha
        test_labels_similar_met = test_labels.query(
            'abs(galah_feh - @self.row1.galah_feh)<0.2 & \
            abs(galah_alpha - @self.row1.galah_alpha)<0.1 & \
            source_id != @self.row1.source_id')

        # if there is no similar star, we can make a q=1 binary
        if len(test_labels_similar_met)==0:
        	test_labels_similar_met = test_labels.query(
            'abs(galah_feh - @self.row1.galah_feh)<0.25 & \
            abs(galah_alpha - @self.row1.galah_alpha)<0.1')

        self.row2 = test_labels_similar_met.sample().iloc[0]

        # assert teff1>teff2
        if self.row1.galah_teff<self.row2.galah_teff:
            row_temp = self.row2
            self.row2 = self.row1
            self.row1 = row_temp
        
        # simulate spectrum
        # compute rv shift
        rv1 = 0
        rv2 = np.random.uniform(0,10)
        # compute relative fluxes
        flux1_weight, flux2_weight = custom_model.flux_weights(
            self.row1.galah_teff, 
            self.row2.galah_teff)
        flux1, sigma1 = test_flux[str(self.row1.source_id)], test_sigma[str(self.row1.source_id)]
        flux2, sigma2 = test_flux[str(self.row2.source_id)], test_sigma[str(self.row2.source_id)]
        # shift flux2 according to drv
        delta_w1 = custom_model.w * rv1/custom_model.speed_of_light_kms
        delta_w2 = custom_model.w * rv2/custom_model.speed_of_light_kms
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
        
        # compute chi-squared of best-fit model
        self.binary_fit_labels, self.binary_fit_chisq = custom_model.fit_binary(
            self.flux, 
            self.sigma_ca_mask)
        
        # compute chi-squared of binary model with true labels
        true_param1 = self.row1[custom_model.training_labels].values.tolist() + [rv1]
        true_param2 = self.row2[['galah_teff','galah_logg','galah_vbroad']].tolist() + [rv2]
        self.true_binary_model = custom_model.binary_model(true_param1, true_param2)
        weights = 1/np.sqrt(self.sigma_ca_mask**2+custom_model.single_star_model.s2)
        resid = weights * (self.true_binary_model - self.flux)
        self.true_binary_model_chisq = np.sum(resid**2)  
        
n_global_min_not_found = 0  
for i in range(100):
	print(i)
	sim_binary = SemiEmpiricalBinarySpectrum()
	if sim_binary.binary_fit_chisq > sim_binary.true_binary_model_chisq:
	        n_global_min_not_found +=1
            
print('{}/100 fits to semi-empirical binaries did not converge \
on a global minimum with true values'.format(n_global_min_not_found)) 