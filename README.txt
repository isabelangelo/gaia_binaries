This file describes the workflow and individual files.

(1) load_training_and_test_data.py loads the GALAH sample, filters, and divides into training/test data 
that gets written to files:
	- data/galah_catalogs stores the catalogs that are loaded to make the training set
	- data/label_dataframes stores the labels for different star/binary samples
	- data/cannon_training_data stores fits files with training flux, sigma loaded to train the cannon
	- data/gaia_rvs_dataframes stores the flux and sigma of different star/binary samples in dataframes
	  (labeled by source_id)
(2) train_cannon_model.py trains the cannon and saves the trained models to data/cannon_models