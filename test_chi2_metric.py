from load_data import *
from chi2_metric import *
import matplotlib.pyplot as plt

# load data spectrum
#data_filename = '../kepler_binaries/gaia_fete/1598406168363183872/RVS_spectrum.xml'
data_filename = './test_binaries/Traven2020/5813134163800075520.xml'
wl, flux, flux_err = load_spectrum(data_filename)

# load doppleganger spectrum
#doppleganger_filedir = '../kepler_binaries/gaia_fete/1598406168363183872/avg_spectra/'
doppleganger_filedir = './test_binaries/Traven2020/5813134163800075520_dopplegangers/'
model_flux, model_flux_err, model_flux_arr, model_flux_err_arr = load_doppleganger_model(doppleganger_filedir)

wl_min_850, wl_max_850 = 849.5,851.0
wl_min_854, wl_max_854 = 853.5,855.5
wl_min_865, wl_max_865 = 865.5,867.5


model_chi2_850, data_chi2_850, delta_chi2_850 = delta_chi_squared(
	flux,  
	flux_err, 
	model_flux, 
	model_flux_err,  
	model_flux_arr,
	model_flux_err_arr,
	wl_min=wl_min_850,
	wl_max=wl_max_850)

model_chi2_854, data_chi2_854, delta_chi2_854 = delta_chi_squared(
	flux,  
	flux_err, 
	model_flux, 
	model_flux_err,  
	model_flux_arr,
	model_flux_err_arr,
	wl_min=wl_min_854,
	wl_max=wl_max_854)

model_chi2_865, data_chi2_865, delta_chi2_865 = delta_chi_squared(
	flux,  
	flux_err, 
	model_flux, 
	model_flux_err,  
	model_flux_arr,
	model_flux_err_arr,
	wl_min=wl_min_865,
	wl_max=wl_max_865)

fig = plt.figure(figsize=(10,8))
ax1 = plt.subplot2grid(shape=(3, 3), loc=(0, 0), colspan=3)
ax2 = plt.subplot2grid(shape=(3, 3), loc=(1, 0), colspan=3)
ax3 = plt.subplot2grid((3, 3), (2, 0))
ax4 = plt.subplot2grid((3, 3), (2, 1))
ax5 = plt.subplot2grid((3, 3), (2, 2))

ax1.errorbar(wl, flux, yerr=flux_err, 
             fmt='-', color='k', ecolor='lightgrey', zorder=0, lw=2, elinewidth=3)
ax1.plot(wl, model_flux, lw=2, color='tomato')
ax1.set_ylabel('normalized flux');plt.xlabel('wavelength (nm)')
ax1.axvspan(wl_min_850, wl_max_850, alpha=0.2, color='tomato')
ax1.axvspan(wl_min_854, wl_max_854, alpha=0.2, color='tomato')
ax1.axvspan(wl_min_865, wl_max_865, alpha=0.2, color='tomato')
ax1.set_xlabel('wavelength (nm)')
ax1.set_title(data_filename)

ax2.plot(wl, flux-model_flux,'k-',lw=2)
ax2.set_ylabel('residuals');plt.xlabel('wavelength (nm)')
ax2.set_xticks([])

ax3.hist(model_chi2_850,bins=25, color='dimgrey')
ax3.axvline(data_chi2_850,color='#9E8EE5', lw=4)
ax3.set_title(r'850 nm, $\Delta \chi^2={}$'.format(int(delta_chi2_850)))
ax3.set_xlabel('single star model chi-squared')

ax4.hist(model_chi2_854,bins=25, color='dimgrey')
ax4.axvline(data_chi2_854,color='#9E8EE5', lw=4)
ax4.set_title(r'854 nm, $\Delta \chi^2={}$'.format(int(delta_chi2_854)))
ax4.set_xlabel('single star model chi-squared')

ax5.hist(model_chi2_865,bins=25, color='dimgrey')
ax5.axvline(data_chi2_865,color='#9E8EE5', lw=4)
ax5.set_title(r'865 nm, $\Delta \chi^2={}$'.format(int(delta_chi2_865)))
ax5.set_xlabel('single star model chi-squared')

plt.savefig('/Users/isabelangelo/Desktop/GALAH_binary.png',dpi=300)
