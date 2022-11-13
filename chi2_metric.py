from load_data import *

# load wavelength array 
data_filename = '../kepler_binaries/gaia_fete/1598406168363183872/RVS_spectrum.xml'
wl,_,_ = load_spectrum(data_filename)


def chi_squared(
	flux, 
	flux_err, 
	model_flux, 
	model_flux_err, 
	wl_min=wl.min(), 
	wl_max=wl.max()
	):

    # compute indices in specified wavelength range
    calc_idx = (wl>=wl_min) & (wl<=wl_max)

    # snip flux, model arrays for calculation
    flux = flux[calc_idx]
    flux_err = flux_err[calc_idx]
    model_flux = model_flux[calc_idx]
    model_flux_err = model_flux_err[calc_idx]
    
    # compute chi2
    resid2_norm = (flux - model_flux)**2/(flux_err**2+model_flux_err**2)
    chi2 = np.nansum(resid2_norm)
    return chi2


def delta_chi_squared(
	flux,  
	flux_err, 
	model_flux, 
	model_flux_err, 
	model_flux_arr,
	model_flux_err_arr,
	wl_min=wl.min(), 
	wl_max=wl.max()
	):

	# store all doppleganger chi2
	model_chi2_values = []
	for i in range(len(model_flux_arr)):
	    model_flux_i = model_flux_arr[i]
	    model_flux_err_i = model_flux_err_arr[i]
	    chi2_i = chi_squared(
	    	model_flux_i, 
	    	model_flux_err_i, 
	    	model_flux, 
	    	model_flux_err, 
	    	wl_min=wl_min, 
	    	wl_max=wl_max)
	    model_chi2_values.append(chi2_i)

	# compute data chi2
	data_chi2 = chi_squared(
		flux, 
		flux_err, 
		model_flux, 
		model_flux_err,
		wl_min=wl_min,
		wl_max=wl_max)

	# compute chi2 difference
	delta_chi2 = data_chi2-np.median(model_chi2_values)
	return model_chi2_values, data_chi2, delta_chi2









