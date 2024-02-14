"""
contains single star model (with Calcium mask) and binary cannon model,
and functions to fit model to data
"""
from custom_model_supplementary_functions import *
from scipy.optimize import leastsq
import lmfit
from astropy.io import fits

# function to call binary model
def binary_model(param1, param2, return_components=False, single_star_model=recent_model_version):
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
def fit_single_star(flux, sigma, single_star_model=recent_model_version):
	"""
	Args:
	    flux (np.array): normalized flux data
	    sigma (np.array): flux error data
	"""

	# mask out calcium triplet
	sigma_for_fit = sigma.copy()
	sigma_for_fit[ca_mask] = np.inf

	# function to calculate penalty for low training density
	training_data = single_star_model.training_set_labels
	training_density_kde = stats.gaussian_kde(training_data.T)
	def density_chisq_inflation(param):
		# require finite parameters
		if False in np.isfinite(param):
			return np.inf
		else:
			density = training_density_kde(param)[0]
			if density > 1e-7:
				return 1
			else:
				return np.sqrt((1+np.log10(1e-7/density)))

	# single star model goodness-of-fit
	def residuals(param):
		# re-parameterize from log(vbroad) to vbroad for Cannon
		cannon_param = param.copy()
		cannon_param[-1] = 10**cannon_param[-1]
		# compute chisq
		model = single_star_model(cannon_param)
		weights = 1/np.sqrt(sigma_for_fit**2+single_star_model.s2)
		resid = weights * (model - flux)

		# inflate chisq if labels are in low density label space
		density_weight = density_chisq_inflation(cannon_param)
		return resid * density_weight

	# initial labels from cannon model
	initial_labels = single_star_model._fiducials.copy()
	# re-parameterize from vbroad to log(vroad) in optimizer
	initial_labels[-1] = np.log10(initial_labels[-1]) 
	# perform fit
	fit_labels = leastsq(residuals,x0=initial_labels)[0]
	chi2_fit = np.sum(residuals(fit_labels)**2)
	# re-parameterize from log(vbroad) to vbroad
	fit_cannon_labels = fit_labels.copy()
	fit_cannon_labels[-1] = 10**fit_cannon_labels[-1]
	#print(fit_labels[-1], fit_cannon_labels[-1])
	return fit_cannon_labels, chi2_fit


# fit binary
def fit_binary(flux, sigma, single_star_model=recent_model_version):
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

	# function to calculate penalty for low training density
	training_data = single_star_model.training_set_labels
	training_density_kde = stats.gaussian_kde(training_data.T)
	def density_chisq_inflation(param):
		# require finite parameters
		if False in np.isfinite(param):
			return np.inf
		else:
			density = training_density_kde(param)[0]
			if density > 1e-7:
				return 1
			else:
				return np.sqrt((1+np.log10(1e-7/density)))

	# binary model goodness-of-fit
	def residuals(params):
		# store primary, secondary parameters
		cannon_param1 = params[:6].copy()
		cannon_param2 = params[6:].copy()

		# re-parameterize from log(vbroad) to vbroad for Cannon
		cannon_param1[-2] = 10**cannon_param1[-2]
		cannon_param2[-2] = 10**cannon_param2[-2]
		cannon_param2_full = np.concatenate((cannon_param2[:2],cannon_param1[2:4],cannon_param2[2:3]))

		# prevent model from regions where flux ratio can't be interpolated
		if 2450>cannon_param1[0] or 34000<cannon_param1[0]:
			return np.inf*np.ones(len(flux))
		elif 2450>cannon_param2[0] or 34000<cannon_param2[0]:
			return np.inf*np.ones(len(flux))
		else:
			# compute chisq
			model = binary_model(cannon_param1, cannon_param2, single_star_model=single_star_model)
			weights = 1/np.sqrt(sigma_for_fit**2+single_star_model.s2)
			resid = weights * (flux - model)

			# inflate chisq if labels are in low density label space
			primary_density_weight = density_chisq_inflation(cannon_param1[:-1])
			secondary_density_weight = density_chisq_inflation(cannon_param2_full)
			density_weight = primary_density_weight * secondary_density_weight
			return resid * density_weight

	fiducial_params = recent_model_version._fiducials
	
	def optimizer(initial_teff):
		# determine initial labels
		rv_i = 0
		teff1_i, teff2_i = initial_teff
		logg_i, feh_i, alpha_i, vbroad_i = single_star_model._fiducials[1:]
		# re-parameterize from vbroad to log(vroad) for optimizer
		logvbroad_i = np.log10(vbroad_i)
		initial_labels = [teff1_i, logg_i, feh_i, alpha_i, logvbroad_i, rv_i, 
						teff2_i, logg_i, logvbroad_i, rv_i]

		# perform least-sqaures fit
		fit_labels = leastsq(residuals,x0=initial_labels)[0]
		chi2_fit = np.sum(residuals(fit_labels)**2)
		# re-parameterize from log(vbroad) to vbroad
		fit_cannon_labels = fit_labels.copy()
		fit_cannon_labels[4] = 10**fit_cannon_labels[4]
		fit_cannon_labels[8] = 10**fit_cannon_labels[8]
		return fit_cannon_labels, chi2_fit

	# initial_teff_arr = [(4100,4000), (6000,4000), (8000,4000), 
 #                   (6100,6000), (8000,6000), (8000,7900)]

	initial_teff_arr = [(4100,4000), (6000,4000), (8000,4000), 
	                   (6100,6000), (8000,6000), (8000,7900),
	                   (5000,4000), (7000,4000), (7000,5000), 
	                   (7000,6000), (6000,5000), (5000,5000),
	                   (7000,7000), (8000,5000), (8000,7000)]

	# run optimizers, store fit with lowest chi2
	lowest_global_chi2 = np.inf    
	best_fit_labels = None

	for initial_teff in initial_teff_arr:
		results = optimizer(initial_teff)
		#print(results[0][0], results[0][6], results[1])
		if results[1] < lowest_global_chi2:
		    lowest_global_chi2 = results[1]
		    best_fit_labels = np.array(results[0])
	return best_fit_labels, lowest_global_chi2



