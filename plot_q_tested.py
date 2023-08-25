import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

df = pd.read_csv('q_tested.txt', names=['q',',','chisq', 'logg'], delim_whitespace=True)

plt.scatter(df.q, df.logg, c=np.log10(df.chisq));
plt.colorbar(label='log(chi2)');plt.xlim(0.1);
plt.ylim(3.9,5);plt.xlabel('sampled q');plt.ylabel('sampled logg');
plt.show()