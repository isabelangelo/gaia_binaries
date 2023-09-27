import custom_model
import gaia_spectrum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# compute metrics for sample of semi-empirical synthetic binaries
keys = ['delta_chisq', 'f_imp', 'true_q', 'true_teff1', 'true_drv']
sim_spec_data = []
for i in range(5):
    sim_spec = gaia_spectrum.SemiEmpiricalBinarySpectrum()
    sim_spec.compute_binary_detection_stats()
    sim_spec_values = [sim_spec.delta_chisq, sim_spec.f_imp, \
              sim_spec.true_q, sim_spec.true_teff1, sim_spec.true_drv]
    sim_spec_data.append(dict(zip(keys, sim_spec_values)))
sim_spec_df = pd.DataFrame(sim_spec_data)

# compute metrics for single stars from El-Badry 2018
single_data = []
for source_id in gaia_spectrum.single_labels.source_i[:5]:
    flux = gaia_spectrum.single_flux[str(source_id)]
    sigma = gaia_spectrum.single_sigma[str(source_id)]
    spec = gaia_spectrum.GaiaSpectrum(
        source_id, 
        flux, 
        sigma, 
        model_to_use = custom_model.recent_model_version)
    single_data.append(dict(zip(['delta_chisq'], [spec.delta_chisq])))
single_df = pd.DataFrame(single_data)

# save data to file
sim_spec_df.to_csv('./data/binary_metric_dataframes/semi_empirical_binary_metrics.csv')
single_df.to_csv('./data/binary_metric_dataframes/single_metrics.csv')

# generate plots
# figure B1
plt.rcParams['font.size']=15
plt.figure(figsize=(15,8))
plt.subplot(121)
plt.scatter(sim_spec_df.true_q, np.log10(sim_spec_df.delta_chisq), c=sim_spec_df.true_drv,
       marker='o', ec='k', s=50, cmap='Reds')
plt.colorbar(location='top', pad=0, label=r'$\Delta$RV (km/s)')
plt.xlabel('q=m2/m1');plt.ylabel(r'log($\chi^2_{\rm single}$-$\chi^2_{\rm binary}$)')
plt.subplot(122)
plt.scatter(sim_spec_df.true_q, np.log10(sim_spec_df.delta_chisq), c=sim_spec_df.true_teff1,
       marker='o', ec='k', s=50, cmap='cool')
plt.colorbar(location='top', pad=0, label=r'$T_{\rm eff}$ of primary')
plt.xlabel('q=m2/m1')
plt.savefig('/Users/isabelangelo/Desktop/figure_B1.png', dpi=300)

# figure B2
plt.plot(sim_spec_df.f_imp, np.log10(sim_spec_df.delta_chisq), 'o', color='k')
plt.xlabel(r'$f_{\rm imp}$')
plt.ylabel(r'log($\chi^2_{\rm single}$-$\chi^2_{\rm binary}$)')
plt.savefig('/Users/isabelangelo/Desktop/figure_B2.png', dpi=300)

# figure B3
plt.hist(np.log10(single_df.delta_chisq), color='k', histtype='step')
plt.hist(np.log10(sim_spec_df.delta_chisq), color='r', histtype='step')
plt.xlabel(r'log($\chi^2_{\rm single}$-$\chi^2_{\rm binary}$)')
ply.ylabel('count')
plt.savefig('/Users/isabelangelo/Desktop/figure_B3.png', dpi=300)