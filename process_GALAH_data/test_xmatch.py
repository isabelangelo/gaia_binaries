from astropy.table import Table
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astroquery
from astroquery.vizier import Vizier

# load single star catalogs
galah_star_catalog = Table.read('./GALAH_data_tables/GALAH_DR3_main_allstar_v2.fits', format='fits').to_pandas()
galah_star_catalog_with_designations = pd.read_csv('./GALAH_data_tables/GALAH_star_catalog.csv')
gaia_star_catalog = pd.read_csv('./GALAH_data_tables/GALAH_stars-result.csv')
gaia_stars_filtered = pd.read_csv('./GALAH_data_tables/GALAH_stars_filtered.csv')
# combine training and test set
training_set = pd.read_csv('../train_cannon_model/data_files/cannon_training_set.csv')
test_set = pd.read_csv('../train_cannon_model/data_files/cannon_test_set.csv')
training_and_test = pd.concat((training_set, test_set))

# load binary catalogs
catalogs = Vizier.find_catalogs('Traven')
catalogs = {k: v for k, v in catalogs.items() if 'J/A+A/638/A145' in k}
Vizier.ROW_LIMIT = -1
catalogs = Vizier.get_catalogs(catalogs.keys())
galah_binary_catalog = catalogs[0].to_pandas()
gaia_binary_catalog = pd.read_csv('../process_GALAH_data/GALAH_data_tables/GALAH_binaries-result.csv')
gaia_binaries_filtered = pd.read_csv('../process_GALAH_data/GALAH_data_tables/GALAH_binaries_filtered.csv')

# function to extract row value
def rvalue(row_column):
    return row_column.to_numpy()[0]

# ==================== tests for single star sample ======================================== 
print('Testing single star catalog cross-match...')
# test 1: check GALAH + Gaia single star samples
# first verify that the ra, dec from the GALAH and gaia catalogs agree for each target
galah_catalog_gaia_catalog_matched = True
merged_sobject_ids = pd.merge(galah_star_catalog, gaia_star_catalog, on='sobject_id')
if np.max(merged_sobject_ids.ra - merged_sobject_ids.ra_dr2) > 0.1:
	galah_catalog_gaia_catalog_matched = False
	print('GALAH star catalog vs. Gaia star catalog: found mismatched ra pair')
if np.max(merged_sobject_ids.dec - merged_sobject_ids.dec_dr2) > 0.1:
	print('GALAH star catalog vs. Gaia star catalog: found mismatched dec pair')
	galah_catalog_gaia_catalog_matched = False
# then verify that the sanme souce_id/designation pairs are in the star catalog
if merged_sobject_ids[['sobject_id','designation']].equals(
	galah_star_catalog_with_designations[['sobject_id','designation']]) == False:
	galah_catalog_gaia_catalog_matched = False

if galah_catalog_gaia_catalog_matched:
	print('test 1 passed, galah and gaia catalogs are well-matched')
else:
	print('WARNING: test 1 failed')


# test 2: check Gaia single stars + training/test sets
gaia_catalog_gaia_filtered_matched = True
for i in range(len(gaia_stars_filtered)):
    filtered_row = gaia_stars_filtered.iloc[i]
    catalog_row = gaia_star_catalog[gaia_star_catalog.sobject_id==filtered_row.sobject_id]
    
    filtered_tuple = (filtered_row.sobject_id, filtered_row.designation, filtered_row.source_id)
    catalog_tuple = rvalue(catalog_row.sobject_id), rvalue(catalog_row.designation), rvalue(catalog_row.source_id)
    
    if filtered_tuple != catalog_tuple:
        print('Gaia star catalog vs. Gaia filtered stars:')
        print('mismatched (sobject_id, designation, source_id)')
        print('Gaia star catalog:{}'.format(catalog_tuple))
        print('Gaia filtered stars:{}'.format(filtered_tuple))
        gaia_catalog_gaia_filtered_matched = False

if gaia_catalog_gaia_filtered_matched:
    print('test 2 passed, gaia catalog and filtered gaia stars are well-matched')


gaia_catalog_training_test_matched = True
for i in range(len(training_and_test)):
    training_test_row = training_and_test.iloc[i]
    filtered_row = gaia_stars_filtered[gaia_stars_filtered.sobject_id==training_test_row.sobject_id]
    
    training_test_tuple = (training_test_row.sobject_id, training_test_row.designation, training_test_row.source_id)
    filtered_tuple = (rvalue(filtered_row.sobject_id),rvalue(filtered_row.designation),rvalue(filtered_row.source_id))        
    
    if training_test_tuple != filtered_tuple:
        print('Gaia filtered stars vs. training/test set:')
        print('mismatched (sobject_id, designation, source_id)')
        print('Gaia star catalog:{}'.format(catalog_tuple))
        print('Gaia filtered stars:{}'.format(filtered_tuple))
        gaia_catalog_training_test_matched = False

if gaia_catalog_training_test_matched:
    print('test 3 passed, gaia filtered stars and training/test set are well-matched')  


# ==================== tests for binary sample ======================================== 
print('Testing binary catalog cross-match...')
# test 1: check GALAH + Gaia single star samples
galah_catalog_gaia_catalog_matched_binaries = True
for i in range(len(gaia_binary_catalog)):
    gaia_row = gaia_binary_catalog.iloc[i]
    galah_row = galah_binary_catalog[galah_binary_catalog.spectID==gaia_row.sobject_id]
    
    if np.abs(rvalue(galah_row.RAJ2000 - gaia_row.ra))>0.1:
        print('GALAH star catalog vs. Gaia star catalog: found mismatched ra pair')
        print('mismatched on row for sobject_id {}'.format(sobject_id))
        galah_catalog_gaia_catalog_matched_binaries = False
    if np.abs(rvalue(galah_row.DEJ2000 - gaia_row.dec))>0.1:
        print('GALAH star catalog vs. Gaia star catalog: found mismatched dec pair')
        print('mismatched on row for sobject_id {}'.format(sobject_id))
        galah_catalog_gaia_catalog_matched_binaries = False
if galah_catalog_gaia_catalog_matched_binaries:
    print('test 1 passed, galah and gaia binary catalogs are well-matched')

# test 2: check Gaia single stars + training/test sets
gaia_catalog_gaia_filtered_matched_binaries = True
for i in range(len(gaia_binaries_filtered)):
    filtered_row = gaia_binaries_filtered.iloc[i]
    catalog_row = gaia_binary_catalog[gaia_binary_catalog.sobject_id==filtered_row.sobject_id]
    
    filtered_tuple = (filtered_row.sobject_id, filtered_row.designation, filtered_row.source_id)
    catalog_tuple = rvalue(catalog_row.sobject_id), rvalue(catalog_row.designation), rvalue(catalog_row.source_id)
    
    if filtered_tuple != catalog_tuple:
        print('Gaia star catalog vs. Gaia filtered stars:')
        print('mismatched (sobject_id, designation, source_id)')
        print('Gaia star catalog:{}'.format(catalog_tuple))
        print('Gaia filtered stars:{}'.format(filtered_tuple))
        gaia_catalog_gaia_filtered_matched_binaries = False
if galah_catalog_gaia_catalog_matched_binaries:
    print('test 2 passed, gaia catalog and filtered gaia stars are well-matched')


# test 3: check sobject IDs + spectIDs still match
if False in (gaia_binaries_filtered['spectID']==gaia_binaries_filtered['sobject_id']):
    print('spectID and sobject_id mismatched in filtered binary sample')
else:
    print('test 3 passed: spectID and sobject_id match in final filtered sample')





