import custom_model
import gaia_spectrum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# this should save the right metrics, let me make new dataframes 
# with smaller sizes before I run the entire thing
# problem: the primary and secondary are switched sometimes
# how can I assert that they're not

# compute metrics for sample of semi-empirical synthetic binaries
sim_spec_keys = [
    'delta_chisq', 
    'f_imp', 
    'teff1_true', 
    'logg1_true', 
    'feh_true',
    'q_true', 
    'drv_true' , 
    'teff1_cannon', 
    'logg1_cannon', 
    'feh_cannon',
    'q_cannon', 
    'drv_cannon']

sim_spec_data = []
for i in range(500):
    sim_spec = gaia_spectrum.SemiEmpiricalBinarySpectrum()
    sim_spec.compute_binary_detection_stats()
    sim_spec_values = [
    sim_spec.delta_chisq, 
    sim_spec.f_imp, 
    sim_spec.row1.teff_gspphot, 
    sim_spec.row1.logg_gspphot, 
    sim_spec.row1.mh_gspphot,
    sim_spec.q_true, 
    sim_spec.rv2 - sim_spec.rv1, 
    sim_spec.primary_fit_labels[0], # cannon teff1
    sim_spec.primary_fit_labels[1], # cannon logg1
    sim_spec.primary_fit_labels[2], # cannon feh1
    sim_spec.q_cannon, 
    sim_spec.secondary_fit_labels[-1] - sim_spec.primary_fit_labels[-1]] # cannon drv
    sim_spec_data.append(dict(zip(sim_spec_keys, sim_spec_values)))
sim_spec_df = pd.DataFrame(sim_spec_data)

# compute metrics for single stars from El-Badry 2018
single_keys = ['delta_chisq', 'f_imp']
single_data = []
for source_id in gaia_spectrum.single_labels.source_id:
    flux = gaia_spectrum.single_flux[str(source_id)]
    sigma = gaia_spectrum.single_sigma[str(source_id)]
    spec = gaia_spectrum.GaiaSpectrum(
        source_id, 
        flux, 
        sigma, 
        model_to_use = custom_model.recent_model_version)
    single_values = [spec.delta_chisq, spec.f_imp]
    single_data.append(dict(zip(single_keys, single_values)))
single_df = pd.DataFrame(single_data)

# save data to file
sim_spec_df.to_csv('./data/binary_metric_dataframes/semi_empirical_binary_metrics_v2.csv')
single_df.to_csv('./data/binary_metric_dataframes/single_metrics_v2.csv')

# generate plots
# figure B1
plt.rcParams['font.size']=15
plt.figure(figsize=(15,8))
plt.subplot(121)
plt.scatter(sim_spec_df.q_true, np.log10(sim_spec_df.delta_chisq), c=sim_spec_df.drv_true,
       marker='o', ec='k', s=50, cmap='Reds')
plt.colorbar(location='top', pad=0, label=r'$\Delta$RV (km/s)')
plt.xlabel('q=m2/m1');plt.ylabel(r'log($\chi^2_{\rm single}$-$\chi^2_{\rm binary}$)')
plt.subplot(122)
plt.scatter(sim_spec_df.q_true, np.log10(sim_spec_df.delta_chisq), c=sim_spec_df.teff1_true,
       marker='o', ec='k', s=50, cmap='cool')
plt.colorbar(location='top', pad=0, label=r'$T_{\rm eff}$ of primary')
plt.xlabel('q=m2/m1')
plt.savefig('/Users/isabelangelo/Desktop/figure_B1.png', dpi=300)

# figure B2
plt.figure(figsize=(8,8))
plt.plot(sim_spec_df.f_imp, np.log10(sim_spec_df.delta_chisq), 'o', color='k')
plt.xlabel(r'$f_{\rm imp}$')
plt.ylabel(r'log($\chi^2_{\rm single}$-$\chi^2_{\rm binary}$)')
plt.savefig('/Users/isabelangelo/Desktop/figure_B2.png', dpi=300)

# figure B3
plt.figure(figsize=(8,8))
plt.hist(np.log10(single_df.delta_chisq), color='k', histtype='step')
plt.hist(np.log10(sim_spec_df.delta_chisq), color='r', histtype='step')
plt.xlabel(r'log($\chi^2_{\rm single}$-$\chi^2_{\rm binary}$)')
plt.ylabel('count')
plt.savefig('/Users/isabelangelo/Desktop/figure_B3.png', dpi=300)