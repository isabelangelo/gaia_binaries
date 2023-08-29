"""
contains single star model (with Calcium mask) and binary cannon model,
and functions to fit model to data
"""
import thecannon as tc
from astropy.table import Table
from custom_model_supplementary_functions import *
from scipy.optimize import leastsq
from astropy.io import fits

# # remove warnings that come from calcium mask
# pd.options.mode.chained_assignment = None 

# =====================================================================================
# load cannon model to use (this needs to be changed to test different models)
single_star_model = tc.CannonModel.read('./data/cannon_models/gaia_rvs_model.model')
model_fileroot = 'gaia_rvs_model'

training_labels = ['galah_teff', 'galah_logg','galah_feh', 'galah_alpha', 'galah_vbroad']
training_set_table = Table.read('./data/label_dataframes/training_labels.csv', format='csv')
training_set = training_set_table[training_labels]

training_flux_df = pd.read_csv('./data/gaia_rvs_dataframes/training_flux.csv')
training_sigma_df = pd.read_csv('./data/gaia_rvs_dataframes/training_sigma.csv')

# ======================================================================================

# load single star cannon model + wavelength
w = fits.open('./data/cannon_training_data/gaia_rvs_wavelength.fits')[0].data[20:-20]

# # broad mask
# ca_idx1 = np.where((w>849) & (w<851))[0]
# ca_idx2 = np.where((w>853.5) & (w<855.5))[0]
# ca_idx3 = np.where((w>865.5) & (w<867.5))[0]

# this is a narrow mask
ca_idx1 = np.where((w>849.5) & (w<850.5))[0]
ca_idx2 = np.where((w>854) & (w<855))[0]
ca_idx3 = np.where((w>866) & (w<867))[0]

# function to calculate penalty for low training density
def density_chisq_inflation(param):
	density = training_density(param)
	if density > 1e-7:
		return 1
	else:
		return (1+np.log10(1e-7/density))

ca_mask = np.array(list(ca_idx1) + list(ca_idx2) + list(ca_idx3))

# function to call binary model
def binary_model(param1, param2, return_components=False):
	"""
	param1 : teff, logg, feh, alpha, vbroad and RV of primary
	param2 : teff, logg, vbroad and RV of secondary
	note: feh, alpha of secondary are assumed to be the same as the primary
	"""

	# store primary, secondary labels
	teff1, logg1, feh1, alpha1, vbroad1, rv1 = param1
	teff2, logg2, vbroad2, rv2 = param2

	# assume same metallicity for both components
	feh2, alpha2 = feh1, alpha1 

	# compute single star models for both components
	flux1 = single_star_model([teff1, logg1, feh1, alpha1, vbroad1])
	flux2 = single_star_model([teff2, logg2, feh2, alpha2, vbroad2])

	# shift flux2 according to drv
	delta_w1 = w * rv1/speed_of_light_kms
	delta_w2 = w * rv2/speed_of_light_kms
	flux1_shifted = np.interp(w, w + delta_w1, flux1)
	flux2_shifted = np.interp(w, w + delta_w2, flux2)


	# compute relative flux based on spectral type
	flux1_weight, flux2_weight = flux_weights(teff1, teff2)

	# add weighted spectra together
	flux_combined = flux1_weight*flux1_shifted + flux2_weight*flux2_shifted

	if return_components:
		return flux1_weight*flux1_shifted, flux2_weight*flux2_shifted, flux_combined
	else:
		return flux_combined


# fit single star
def fit_single_star(flux, sigma, mask_calcium=True, training_density_minimum=True):
	"""
	Args:
		flux (np.array): normalized flux data
		sigma (np.array): flux error data
	"""

	# mask out calcium triplet
	sigma_for_fit = sigma.copy()
	if mask_calcium:
		sigma_for_fit[ca_mask] = np.inf

	# single star model goodness-of-fit
	def residuals(param):

		# compute chisq
		model = single_star_model(param)

		# original version to compute weights
		weights_old = 1/sigma_for_fit

		# new version to compute weights
		weights = 1/np.sqrt(sigma_for_fit**2+single_star_model.s2)

		resid = weights * (model - flux)

		# inflate chisq if labels are in low density label space
		if training_density_minimum:
			density_weight = density_chisq_inflation(param)
		else:
			density_weight = 1
		return resid * density_weight

	# print('running single star optimizer on 1 spectrum')
	initial_labels = single_star_model._fiducials
	fit_labels = leastsq(residuals,x0=initial_labels)[0]
	chi2_fit = np.sum(residuals(fit_labels)**2)
	return fit_labels, chi2_fit


# fit binary
def fit_binary(flux, sigma):
	"""
	Args:
		flux (np.array): normalized flux data
		sigma (np.array): flux error data
		initial_teff (tuple): initial guess for Teff1, Teff2,
		e.g., (5800, 4700)
	"""

	# mask out calcium triplet
	sigma_for_fit = sigma.copy()
	sigma_for_fit[ca_mask] = np.inf

	# binary model goodness-of-fit
	def residuals(params):

		# store primary, secondary parameters
		param1 = params[:6]
		param2 = params[6:]
		param2_full = np.concatenate((param2[:2],param1[2:4],param2[2:3]))

		# prevent model from regions where flux ratio can't be interpolated
		if 2450>param1[0] or 34000<param1[0]:
			return np.inf*np.ones(len(flux))
		elif 2450>param2[0] or 34000<param2[0]:
			return np.inf*np.ones(len(flux))
		else:
			# compute chisq
			model = binary_model(param1, param2)
			weights = 1/sigma_for_fit
			resid = weights * (flux - model)

			# inflate chisq if labels are in low density label space
			density_weight = density_chisq_inflation(param1[:-1]) * density_chisq_inflation(param2_full)


			# print mass ratio being tested
			# q_to_print = teff2mass(param1[0])/teff2mass(param2[0])
			# if q_to_print>1:
			# 	print(1/q_to_print, ',', np.sum((resid * density_weight)**2), param1[1])
			# else:
			# 	print(q_to_print, ',', np.sum((resid * density_weight)**2), param1[1])
			return resid * density_weight

	# single optimizer
	def optimizer(initial_teff):

		# determine initial labels
		rv_i = 0
		teff1_i, teff2_i = initial_teff
		logg_i, feh_i, alpha_i, vbroad_i = single_star_model._fiducials[1:]
		initial_labels = [teff1_i, logg_i, feh_i, alpha_i, vbroad_i, rv_i, teff2_i, logg_i, vbroad_i, rv_i]

		# perform least-sqaures fit
		fit_labels = leastsq(residuals,x0=initial_labels)[0]
		chi2_fit = np.sum(residuals(fit_labels)**2)
		return fit_labels, chi2_fit

	# find true global minimum
	# initial teff guesses for optimizers
	initial_teff_arr = [(4000,4000), (6000,4000), (8000,4000), 
                        (6000,6000), (8000,6000), (8000,8000)]
	# initial_teff_arr = [(6500,4200), (4200,6500)]

	# run optimizers, store fit with lowest chi2
	lowest_global_chi2 = np.inf    
	best_fit_labels = None

	for initial_teff in initial_teff_arr:
		results = optimizer(initial_teff)
		if results[1] < lowest_global_chi2:
			lowest_global_chi2 = results[1]
			best_fit_labels = results[0]
	return best_fit_labels, lowest_global_chi2


# # fit example binary
# elbadry_binary_flux_df = pd.read_csv('./data/gaia_rvs_dataframes/elbadry_tableE3_binaries_flux.csv')
# elbadry_binary_sigma_df = pd.read_csv('./data/gaia_rvs_dataframes/elbadry_tableE3_binaries_sigma.csv')
# elbadry_binary_label_df = pd.read_csv('./data/label_dataframes/elbadry_tableE3_binaries_labels.csv')

# test_binary_source_id = 4346137540665080832
# test_binary_flux = elbadry_binary_flux_df[str(test_binary_source_id)]
# test_binary_sigma = elbadry_binary_sigma_df[str(test_binary_source_id)]
# fit_binary(test_binary_flux, test_binary_sigma)
