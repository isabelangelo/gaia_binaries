from astropy.io.votable import parse
import glob
import numpy as np


def load_spectrum(filename):
	"""
	load and store Gaia spectrum
	Args:
		filename (str): path to RVS spectrum file
	Returns:
		wl (np.arr): wavelengths (same for all Gaia RVS spectra)
		flux (np.arr): normalized flux
		flux_err (np.arr): flux errors
	"""
	votable =  parse(filename)
	table = votable.get_first_table()
    
	flux = table.array['flux']
	flux_err = table.array['flux_error']
	wl = table.array['wavelength']

	return wl, flux, flux_err


def load_doppleganger_model(filedir):
	"""
	create spectrum model from doppleganger spectra
	Args:
		filedir (str): path to directory containing doppleganger 
			spectrum files
	Returns:
		model_flux (np.arr): averaged normalized flux
		flux_err (np.arr): model flux error
	"""
	model_filenames = glob.glob(filedir+'*.xml')

	model_fluxes=[]
	model_flux_errs=[]

	for model_filename in model_filenames:
	    _,model_flux,model_flux_err = load_spectrum(model_filename) 
	    model_fluxes.append(model_flux)
	    model_flux_errs.append(model_flux_err)
	    
	model_flux_arr = np.array(model_fluxes)
	model_flux_err_arr = np.array(model_flux_errs)
	    
	model_flux = np.nanmean(model_fluxes, axis=0)
	model_flux_err = np.nanmean(model_flux_errs, axis=0)/np.sqrt(len(model_fluxes))

	return model_flux, model_flux_err, model_flux_arr, model_flux_err_arr

