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
		density = training_density_kde(param)[0]
		if density > 1e-7:
			return 1
		else:
			return np.sqrt((1+np.log10(1e-7/density)))

	# single star model goodness-of-fit
	def residuals(param):
	    # compute chisq
	    model = single_star_model(param)
	    weights = 1/np.sqrt(sigma_for_fit**2+single_star_model.s2)
	    resid = weights * (model - flux)

	    # inflate chisq if labels are in low density label space
	    density_weight = density_chisq_inflation(param)
	    return resid * density_weight

	# print('running single star optimizer on 1 spectrum')
	# fit_params = lmfit.Parameters()
	# init_params = single_star_model._fiducials
	# fit_params.add('teff', value=init_params[0], min=4000, max=8000)
	# fit_params.add('logg',value=init_params[1], min=4, max=5)
	# fit_params.add('feh',value=init_params[2], min=-1.5, max=1.5)
	# fit_params.add('alpha',value=init_params[3], min=-1, max=1)
	# fit_params.add('vbroad',value=init_params[4], min=0, max=100)
	# result = lmfit.minimize(residuals, fit_params)
	# fit_labels = np.array([value.value for key, value in result.params.items()])
	# chi2_fit = result.chisqr
	# return fit_labels, chi2_fit

	initial_labels = single_star_model._fiducials
	fit_labels = leastsq(residuals,x0=initial_labels)[0]
	chi2_fit = np.sum(residuals(fit_labels)**2)
	return fit_labels, chi2_fit



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
		density = training_density_kde(param)[0]
		if density > 1e-7:
			return 1
		else:
			return np.sqrt((1+np.log10(1e-7/density)))

	# binary model goodness-of-fit
	def residuals(params):

		# store primary, secondary parameters
		#param12 = [value.value for key, value in params.items()]

		# NOTE: this line below can be removed/replaced with params
		param12 = params 
		param1 = param12[:6]
		param2 = param12[6:]
		param2_full = np.concatenate((param2[:2],param1[2:4],param2[2:3]))

		# prevent model from regions where flux ratio can't be interpolated
		if 2450>param1[0] or 34000<param1[0]:
			return np.inf*np.ones(len(flux))
		elif 2450>param2[0] or 34000<param2[0]:
			return np.inf*np.ones(len(flux))
		else:
			# compute chisq
			model = binary_model(param1, param2, single_star_model=single_star_model)
			weights = 1/np.sqrt(sigma_for_fit**2+single_star_model.s2)
			resid = weights * (flux - model)

			# inflate chisq if labels are in low density label space
			density_weight = density_chisq_inflation(param1[:-1]) * density_chisq_inflation(param2_full)
			return resid * density_weight

	fiducial_params = recent_model_version._fiducials
	#single_fit_params, best_fit_single_chisq = fit_single_star(flux, sigma)
	def optimizer(initial_teff):
		# #print(initial_teff)
		# # print('running single star optimizer on 1 spectrum')
		# fit_params = lmfit.Parameters()
		# fit_params.add('teff1', value=initial_teff[0], min=4000, max=8000)
		# fit_params.add('logg1',value=fiducial_params[1], min=4, max=5)
		# fit_params.add('feh',value=fiducial_params[2], min=-1.5, max=1.5)
		# fit_params.add('alpha',value=fiducial_params[3], min=-1, max=1)
		# fit_params.add('vbroad1',value=fiducial_params[4], min=0, max=100)
		# fit_params.add('rv1', value=0, min=-20, max=20)
		# fit_params.add('teff2', value=initial_teff[1], min=4000, max=8000)
		# fit_params.add('logg2',value=fiducial_params[1], min=4, max=5)
		# fit_params.add('vbroad2',value=fiducial_params[4], min=0, max=100)
		# fit_params.add('rv2', value=0, min=-20, max=20)

		# # perform fit
		# result = lmfit.minimize(residuals, fit_params, method='leastsq')
		# fit_labels = [value.value for key, value in result.params.items()]
		# chi2_fit = result.chisqr
		# return fit_labels, chi2_fit

		# determine initial labels
		rv_i = 0
		teff1_i, teff2_i = initial_teff
		logg_i, feh_i, alpha_i, vbroad_i = single_star_model._fiducials[1:]
		initial_labels = [teff1_i, logg_i, feh_i, alpha_i, vbroad_i, rv_i, teff2_i, logg_i, vbroad_i, rv_i]

		# perform least-sqaures fit
		fit_labels = leastsq(residuals,x0=initial_labels)[0]
		chi2_fit = np.sum(residuals(fit_labels)**2)
		return fit_labels, chi2_fit

	initial_teff_arr = [(4100,4000), (6000,4000), (8000,4000), 
	                   (6100,6000), (8000,6000), (8000,7900)]

	# run optimizers, store fit with lowest chi2
	lowest_global_chi2 = np.inf    
	best_fit_labels = None

	for initial_teff in initial_teff_arr:
		results = optimizer(initial_teff)
		if results[1] < lowest_global_chi2:
		    lowest_global_chi2 = results[1]
		    best_fit_labels = np.array(results[0])
	return best_fit_labels, lowest_global_chi2



