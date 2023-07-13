
import thecannon as tc
from custom_model_supplementary_functions import *
from scipy.optimize import leastsq
from astropy.io import fits

# load single star cannon model + wavelength
w = fits.open('./data/cannon_training_data/gaia_rvs_wavelength.fits')[0].data[20:-20]
single_star_model = tc.CannonModel.read('./data/cannon_models/gaia_rvs_model.model')

# ca_idx1 = np.where((w>849) & (w<851))[0]
# ca_idx2 = np.where((w>853.5) & (w<855.5))[0]
# ca_idx3 = np.where((w>865.5) & (w<867.5))[0]
# ca_idx = list(ca_idx1) + list(ca_idx2) + list(ca_idx3)
# spec_idx = np.array([i for i in range(len(w)) if i not in ca_idx])


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
def fit_single_star(flux, sigma):

	# single star model goodness-of-fit
	def chisq_single(param):
		model = single_star_model(param)
		weights = 1/sigma
		sq_resid = weights * (flux - model)
		return sq_resid

	initial_labels = single_star_model._fiducials
	fit_labels = leastsq(chisq_single,x0=initial_labels)[0]
	return fit_labels, np.sum(chisq_single(fit_labels))


# fit binary
def fit_binary(flux, sigma):

	# binary model goodness-of-fit
	def chisq_binary(params):
		param1 = params[:6]
		param2 = params[6:]
		param2_full = np.concatenate((param2[:2],param1[2:4],param2[2:3]))
		# if 4000>param1[0] or 7000<param1[0]:
		# 	return np.inf*np.ones(len(flux))
		# elif 4000>param2[0] or 7000<param2[0]:
		# 	return np.inf*np.ones(len(flux))
		# else:
		model = binary_model(param1, param2)
		sq_resid = (flux - model)**2/sigma**2
		return sq_resid

	initial_labels = [6000,4.3,-0.03,0.0,7,0,5800,4.2,5,0]
	fit_labels = leastsq(chisq_binary,x0=initial_labels)[0]
	return fit_labels, np.sum(chisq_binary(fit_labels))

	    




