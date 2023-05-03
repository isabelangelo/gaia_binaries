
####### python scripts + descriptions
GALAH_catalog_gaia_xmatch.py : compiles GALAH star and binary catalogs from literature, matches them to Gaia designations and saves to two .csv files:
	(1)GALAH stars + Gaia designations saved to ./GALAH_data_tables/GALAH_star_catalog.csv
	(2) GALAH binaries + Gaia designations saved to ./GALAH_data_tables/GALAH_binary_catalog.csv

filter_GALAH_star_sample.py : filters GALAH star catalog to create training + test samples. Saves table to ./GALAH_data_tables/GALAH_stars_filtered.csv
^this is the table needs to be uploaded to Gaia ADQL 

full_GALAH_Gaia_xmatch_plots.ipynb : creates HR diagrams + label histograms for every spectrum in the GALAH/Gaia crossmatch .fits file from the GALAH website



#### files that can probably be deleted soon #################
GALAH_sample_label_histrogram.ipynb	: creates histograms of training set, this makes more sense in another directory
investigate_GALAH_gaia_xmatch.ipynb : investigates how well the xmatch is working




	



