
from astropy.table import Table
import pandas as pd
import thecannon as tc
from astropy.io import fits
import matplotlib.pyplot as plt

# load cannon model
model = tc.CannonModel.read('./cannon_models/galah_labels_5para_highSNR_precise_labels.model')

# load flux data, training + test sets, cannon model
w = fits.open('./data_files/gaia_rvs_wavelength.fits')[0].data[20:-20]

# load star data
test_set = pd.read_csv('./data_files/cannon_test_set.csv')
training_set = pd.read_csv('./data_files/cannon_training_set.csv')
flux_data = pd.read_csv('./data_files/flux_data.csv')
flux_err_data = pd.read_csv('./data_files/flux_err_data.csv')

def get_labels(label_df, flux_data_df, flux_err_data_df):
    # lists to store labels
    teff_galah, teff_cannon = [], []
    logg_galah, logg_cannon = [], []
    feh_galah, feh_cannon = [], []
    alpha_galah, alpha_cannon = [], []
    vbroad_galah, vbroad_cannon = [], []

    # iterate over each object
    for source_id in label_df.source_id.to_numpy()[:100]:

        # store galah labels
        row = label_df.loc[label_df.source_id==source_id]
        teff_galah.append(row['galah_teff'].to_numpy()[0])
        logg_galah.append(row['galah_logg'].to_numpy()[0])
        feh_galah.append(row['galah_feh'].to_numpy()[0])
        alpha_galah.append(row['galah_alpha_fe'].to_numpy()[0])
        vbroad_galah.append(row['galah_vbroad'].to_numpy()[0])
        
        # fit cannon model
        flux = flux_data_df[str(source_id)].to_numpy()[20:-20]
        ivar = 1/flux_err_data_df[str(source_id)].to_numpy()[20:-20]**2
        result = model.test(flux, ivar)
        teff_fit, logg_fit, feh_fit, alpha_fit, vbroad_fit = result[0][0]
        
        # store cannon labels
        teff_cannon.append(teff_fit)
        logg_cannon.append(logg_fit)
        feh_cannon.append(feh_fit)
        alpha_cannon.append(alpha_fit)
        vbroad_cannon.append(vbroad_fit)

    # store data in dataframe
    data = {'teff_galah':teff_galah, 
            'logg_galah':logg_galah,
            'feh_galah': feh_galah,
            'alpha_galah':alpha_galah,
            'vbroad_galah':vbroad_galah,
            'teff_cannon':teff_cannon, 
            'logg_cannon':logg_cannon,
            'feh_cannon': feh_cannon,
            'alpha_cannon':alpha_cannon,
            'vbroad_cannon':vbroad_cannon}

    label_df = pd.DataFrame(data)
    return label_df

label_df = get_labels(test_set, flux_data, flux_err_data)


plt.figure(figsize=(25,6))

alpha_value = 1
plt.rcParams['font.size']=15
markersize=5
pointcolor='dimgrey'
plt.subplot(151)
plt.plot(label_df.teff_galah, label_df.teff_cannon, '.', color=pointcolor, ms=markersize, alpha=alpha_value)
plt.xlabel('GALAH teff');plt.ylabel('Cannon teff')
plt.plot([3000,9000], [3000,9000], lw=0.7, color='#AA8ED9')


plt.subplot(152)
plt.plot(label_df.logg_galah, label_df.logg_cannon, '.', color=pointcolor, ms=markersize, alpha=alpha_value)
plt.xlabel('GALAH logg');plt.ylabel('Cannon logg')
plt.plot([3,5], [3,5], lw=0.7, color='#AA8ED9')


plt.subplot(153)
plt.plot(label_df.feh_galah, label_df.feh_cannon, '.', color=pointcolor, ms=markersize, alpha=alpha_value)
plt.xlabel('GALAH feh');plt.ylabel('Cannon feh')
plt.plot([-2,2], [-2,2], lw=0.7, color='#AA8ED9')

plt.subplot(154)
plt.plot(label_df.alpha_galah, label_df.alpha_cannon, '.', color=pointcolor, ms=markersize, alpha=alpha_value)
plt.xlabel('GALAH alpha');plt.ylabel('Cannon alpha')
plt.plot([-2,2], [-2,2], lw=0.7, color='#AA8ED9')

plt.subplot(155)
plt.loglog(label_df.vbroad_galah, label_df.vbroad_cannon, '.', color=pointcolor, ms=markersize, alpha=alpha_value)
plt.xlabel('GALAH vbroad');plt.ylabel('Cannon vbroad')
plt.plot([0.1,100], [0.1,100], lw=0.7, color='#AA8ED9')

plt.subplots_adjust(wspace=0.3)
plt.rcParams['figure.dpi']=300

plt.subplots_adjust(wspace=0.2)
plt.show()
#plt.savefig('/Users/isabelangelo/Desktop/test_binary_labels.png', dpi=150)


