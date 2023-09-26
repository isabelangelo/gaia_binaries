import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gaia_spectrum

# set random seed to generate synthetic binary population
np.random.seed(1234)
        
n_global_min_not_found = 0  
for i in range(100):
	print(i)
	sim_binary = SemiEmpiricalBinarySpectrum()
    sim_binary.compute_optimizer_stats()
	if sim_binary.binary_fit_chisq > sim_binary.true_binary_model_chisq:
	        n_global_min_not_found +=1
            
print('{}/100 fits to semi-empirical binaries did not converge \
on a global minimum with true values'.format(n_global_min_not_found)) 