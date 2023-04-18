import pandas as pd
import thecannon as tc
from astropy.io import fits
import matplotlib.pyplot as plt

# load cannon model
model = tc.CannonModel.read('../train_cannon_model/cannon_models/galah_labels_5para_highSNR_precise_labels.model')

# load flux data, training + test sets, cannon model
w = fits.open('../train_cannon_model/data_files/gaia_rvs_wavelength.fits')[0].data[20:-20]

plt.rcParams['figure.dpi']=150
plt.rcParams['font.size']=15

def plot_fit(source_id, flux_data_df, flux_err_data_df):
    s = flux_data_df[source_id][20:-20]
    serr = flux_err_data_df[source_id][20:-20]
    ivar = 1/serr**2
    
    result = model.test(s, ivar)
    fit = result[2][0]['model_flux']
    
    plt.figure(figsize=(15,10))
    plt.subplot(211);plt.title('Gaia DR3 source ID '+source_id)
    plt.plot(w, s, lw=3, color='grey')
    plt.plot(w, fit, color='salmon', lw=3)
    plt.ylabel('flux')
    plt.subplot(212)
    plt.plot(w, s-fit, lw=3, color='grey')
    plt.xlabel('wavelength (nm)')
    plt.ylabel('residuals')