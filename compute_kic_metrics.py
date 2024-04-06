from astropy.table import Table
import gaia
import gaia_spectrum
import numpy as np
import pandas as pd

# load original Gaia-Kepler crossmatch
gaia_kepler_one2one = Table.read('./data/literature_data/gaia_kepler_xmatch/kepler_dr3_good.fits', 
                  format='fits')
print('{} stars in one-to-one Gaia-Kepler cross-match'.format(len(gaia_kepler_one2one)))

# preliminary cuts to ensure main-sequence stars + RVS spectrum
kic_stars_gaia = gaia_kepler_one2one.to_pandas().query('has_rvs==True')
print('{} have RVS spectra'.format(len(kic_stars_gaia)))
kic_stars_gaia = kic_stars_gaia.query('logg>4 & teff>4000 & teff<7000')
print('{} have logg>4, 4000<Teff<7000'.format(len(kic_stars_gaia)))

# re-format to upload to gaia
kic_stars_gaia = kic_stars_gaia.rename(columns={"planet?": "has_planet"})
kic_stars_gaia['tm_designation'] = kic_stars_gaia['tm_designation'].str.decode("utf-8")
kic_stars_gaia['has_planet'] = kic_stars_gaia['has_planet'].str.decode("utf-8")
kic_stars_gaia = kic_stars_gaia.replace({True: 'True', False: 'False'})

# query to run in Gaia
query=f"SELECT kic.kepid, kic.source_id, dr3.rvs_spec_sig_to_noise, kic.rv_nb_transits, \
dr3.radial_velocity, dr3.radial_velocity_error, kic.non_single_star, kic.ruwe, \
kic.tm_designation, kic.kepmag, kic.nconfp, kic.nkoi, kic.has_planet, \
kic.kepler_gaia_mag_diff \
FROM user_iangelo.kic_stars_gaia as kic \
JOIN gaiadr3.gaia_source as dr3 \
    ON dr3.source_id = kic.source_id \
WHERE dr3.rvs_spec_sig_to_noise>50"

# upload KIC table to Gaia 
#gaia.upload_table(kic_stars_gaia, 'kic_stars_gaia')

# download data
kic_stars_gaia_results, kic_flux_df, kic_sigma_df = gaia.retrieve_data_and_labels(query)
nkic = len(kic_stars_gaia_results)
print(len(kic_stars_gaia_results), '{} KIC RVS spectra found with SNR>50'.format(nkic))

# compute metrics
kic_keys = [
    'kepid',
    'source_id',
    'teff',
    'logg',
    'feh',
    'alphafe',
    'vbroad',
    'log_chisq',
    'log_training_density',
    'log_chisq_ca',
    'log_delta_chisq',
    'rvs_spec_sig_to_noise',
    'rv_nb_transits',
    'radial_velocity', 
    'radial_velocity_error', 
    'non_single_star', 
    'ruwe', 
    'nconfp', 
    'nkoi', 
    'has_planet', 
    'kepler_gaia_mag_diff'
]

kic_data = []
for source_id in kic_stars_gaia_results.source_id:
	print(source_id)
	row = kic_stars_gaia_results[kic_stars_gaia_results.source_id==source_id].iloc[0]
	spec = gaia_spectrum.GaiaSpectrum(source_id,
	                             kic_flux_df[source_id],
	                             kic_sigma_df[source_id])
	spec.compute_oddball_metrics()
	spec.compute_best_fit_binary()
	spec.compute_binary_detection_stats()
	kic_values = [
	    row.kepid,
	    row.source_id,
	    spec.single_fit_labels[0],
	    spec.single_fit_labels[1],
	    spec.single_fit_labels[2],
	    spec.single_fit_labels[3],
	    spec.single_fit_labels[4],
	    np.log10(spec.single_fit_chisq),
	    np.log10(spec.single_fit_training_density),
	    np.log10(spec.single_fit_ca_resid),
	    np.log10(spec.delta_chisq),
	    row.rvs_spec_sig_to_noise,
	    row.rv_nb_transits,
	    row.radial_velocity, 
	    row.radial_velocity_error, 
	    row.non_single_star, 
	    row.ruwe, 
	    row.nconfp, 
	    row.nkoi, 
	    row.has_planet, 
	    row.kepler_gaia_mag_diff]
	kic_data.append(dict(zip(kic_keys, kic_values)))
	kic_df = pd.DataFrame(kic_data)
	
# save data to files
kic_df.to_csv('./data/oddball_and_binary_metric_dataframes/kic_metrics.csv')
kic_flux_df.to_csv('./data/gaia_rvs_dataframes/kic_flux_df.csv')
kic_sigma_df.to_csv('./data/gaia_rvs_dataframes/kic_sigma_df.csv')
