
import pandas as pd
import thecannon as tc
from astropy.io import fits
import matplotlib.pyplot as plt

# load cannon model
model = tc.CannonModel.read('./cannon_models/galah_labels_4para_highSNR_precise_labels.model')

# load flux data, training + test sets, cannon model
w = fits.open('./data_files/gaia_rvs_wavelength.fits')[0].data[20:-20]
test_set = pd.read_csv('./data_files/cannon_test_set.csv')
training_set = pd.read_csv('./data_files/cannon_training_set.csv')
flux_data = pd.read_csv('./data_files/flux_data.csv')
flux_err_data = pd.read_csv('./data_files/flux_err_data.csv')


teff_galah, teff_cannon = [], []
logg_galah, logg_cannon = [], []
feh_galah, feh_cannon = [], []
alpha_galah, alpha_cannon = [], []
vbroad_galah, vbroad_cannon = [], []

df_to_plot = test_set
for source_id in df_to_plot.source_id.to_numpy()[:3000]:
    row = df_to_plot.loc[df_to_plot.source_id==source_id]
    teff_galah.append(row['galah_teff'].to_numpy()[0])
    logg_galah.append(row['galah_logg'].to_numpy()[0])
    feh_galah.append(row['galah_feh'].to_numpy()[0])
    alpha_galah.append(row['galah_alpha_fe'].to_numpy()[0])
    vbroad_galah.append(row['galah_vbroad'].to_numpy()[0])
    
    flux = flux_data[str(source_id)].to_numpy()[20:-20]
    ivar = 1/flux_err_data[str(source_id)].to_numpy()[20:-20]**2
    result = model.test(flux, ivar)
    
    #teff_fit, logg_fit, feh_fit, alpha_fit, vbroad_fit = result[0][0]
    teff_fit, logg_fit, feh_fit, vbroad_fit = result[0][0]
    
    teff_cannon.append(teff_fit)
    logg_cannon.append(logg_fit)
    feh_cannon.append(feh_fit)
    #alpha_cannon.append(alpha_fit)
    vbroad_cannon.append(vbroad_fit)


plt.figure(figsize=(20,5))
alpha_value = 0.5
plt.rcParams['font.size']=15
plt.subplot(141)
plt.plot(teff_galah, teff_cannon, '.', color='dimgrey', ms=3, alpha=alpha_value)
plt.xlabel('GALAH teff');plt.ylabel('Cannon teff')
plt.plot([3000,9000], [3000,9000], lw=0.7, color='#AA8ED9')


plt.subplot(142)
plt.plot(logg_galah, logg_cannon, '.', color='dimgrey', ms=3, alpha=alpha_value)
plt.xlabel('GALAH logg');plt.ylabel('Cannon logg')
plt.plot([3,5], [3,5], lw=0.7, color='#AA8ED9')


plt.subplot(143)
plt.plot(feh_galah, feh_cannon, '.', color='dimgrey', ms=3, alpha=alpha_value)
plt.xlabel('GALAH feh');plt.ylabel('Cannon feh')
plt.plot([-2,2], [-2,2], lw=0.7, color='#AA8ED9')

# plt.subplot(154)
# plt.plot(alpha_galah, alpha_cannon, '.', color='dimgrey', ms=3, alpha=alpha_value)
# plt.xlabel('GALAH alpha');plt.ylabel('Cannon alpha')
# plt.plot([-2,2], [-2,2], lw=0.7, color='#AA8ED9')

plt.subplot(144)
plt.loglog(vbroad_galah, vbroad_cannon, '.', color='dimgrey', ms=3, alpha=alpha_value)
plt.xlabel('GALAH vbroad');plt.ylabel('Cannon vbroad')
plt.plot([0.1,100], [0.1,100], lw=0.7, color='#AA8ED9')

plt.subplots_adjust(wspace=0.3)
plt.rcParams['figure.dpi']=300
plt.savefig('/Users/isabelangelo/Desktop/cannon_galah_labels.png')