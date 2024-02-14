This file describes the workflow and individual files.

code files:
(1) training + validating single star cannon model

	gaia.py	: defines functions to query stars and save their labels and RVS spectra to dataframes.
	cannon_model_diagnostics : defines functions to train Cannon model + create diagnostic plots
	load_training_data.py : loads labels + RVS spectra for the Cannon training set, and high SNR spectra for 
	empirical s2 calculation
	train_cannon_model.py : trains a cannon model originally and then iteratively cleans it to remove binaries.

(2) writing binary cannon model

	custom_model.py : contains single star model (with Calcium mask) and binary cannon model, and functions to fit model to data
	custom_model_supplementary_functions.py : defines functions needed for custom model


(3) validating binary cannon model

	load_galah_binaries.py : loads RVS spectra + labels for binaries from Traven et al 2020
	load_galah_control_sample.py : loads RVS spectra + labels for control sample from GALAH		
	load_elbadry2018_data.py : loads RVS spectra + labels for single stars + binaries from El-Badry et al 2018b


data files:
	cannon_training_data : fits files with training set RVS spectra
	cannon_models : single star cannon models + corresponding diagnostic plots
	binary_models : diagnostic plots for different binary model versions + validation samples
	gaia_rvs_dataframes : RVS spectra for training + test + validation samples
	label_dataframes : literature-reported labels for training + test + validation samples
	literature_data : various tables from GALAH catalogs and validation set literature