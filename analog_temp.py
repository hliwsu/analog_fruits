# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 18:36:16 2020

@author: haoli
"""

import pandas as pd
import numpy as np
import scipy as sci
from scipy import signal

a=pd.read_csv('D:\\PY_temp\\sid\\A.csv', ',')
# del a['Loc']
b=pd.read_csv('D:\\PY_temp\\sid\\B.csv', ',')
c=pd.read_csv('D:\\PY_temp\\sid\\C.csv', ',').sort_values(by=['Loc', 'ind']) # loc X var
del c['ind']

# location X time: 30 X 41
t=[i for i in range(0,30)]*41
a_agg=a.T.to_numpy() #variable X location
a_agg=a_agg[1:,]
b_agg=b.T.to_numpy()
b_agg=b_agg[1:,]
c_agg=c.to_numpy()
c_agg=c_agg[:,1:]
c_agg=np.asarray(c_agg)
c_agg.astype(np.float)
trunc_eig_val=0.1
distances=np.zeros(shape=(30,30))
for loc in range(0,30):
    B_j=b_agg[:,loc]
    # a_sub=a.loc[a['Loc'] == loc].iloc[0,1:b.shape[1]].tolist()
    # b_sub=b.loc[b['Location'] == loc].iloc[0,1:b.shape[1]].tolist()
    # c_sub=c.loc[c['Loc'] == loc]
    # c_ssub=c.loc[c['Loc'] == loc].iloc[:,2:c.shape[1]].values
    C_j=c_agg[loc*41:(loc+1)*41,]
    # C_j=signal.detrend(c_ssub)
    C_j=np.array(C_j, dtype=np.float64)
    C_j_sd=C_j.std(axis=0)
    A_prime=a_agg/C_j_sd[:,None]
    B_j_prime=B_j/C_j_sd
    C_j_prime=C_j/C_j_sd
    ## Step 2: principal component analyses on the reference matrix C, and principal components extraction 
    C_j_prime_avg=np.mean(C_j_prime,axis=0)
    m, n = np.shape(C_j_prime)
    C_adj = []
    C_j_prime_p_avg = np.tile(C_j_prime_avg, (m, 1))
    C_adj = C_j_prime - C_j_prime_p_avg
    # calculate the covariate matrix
    covC = np.cov(C_adj.T)   
    # solve its eigenvalues and eigenvectors
    # C_eigen_val, C_eigen_vec=  np.linalg.eig(covC) /np.linalg.eig generates non-reproducible last few vectors of eigenmatrix  
    C_eigen_val, C_eigen_vec=  sci.linalg.eig(covC) #/ using the Hermitian solver, which will generate non-machine-specified results, and the results are also the same with MATLAB's results
    # rank the eigenvalues: in here, I did not apply the truncation rule for the sake of limited variable availability
    index = np.argsort(-C_eigen_val) # equal to index = eigenValues.argsort()[::-1]   
    # apply the truncation rule
    # topn_index = sorted_idx[:n_components]
    # topn_vects = eig_vects[topn_index, :]
    C_eigen_val=C_eigen_val[index]
    C_eigen_vec=C_eigen_vec[:,index]
    C_eigen_val_count=len([ct for ct in C_eigen_val if ct>=trunc_eig_val] )
    finalData = []
    # C matrix, corrected with PCA
    C_pca_vec=C_eigen_vec.T
    # A and B matrices, corrected with PCA
    X=A_prime.T.dot(C_pca_vec)
    Y_j=B_j_prime.T.dot(C_pca_vec)
    # project C to PCA
    Z_j=C_j_prime.dot(C_pca_vec)
    ## Step 3a: standarlization of anomalies 
    Z_j_sd=Z_j.std(axis=0)
    X_prime=X/Z_j_sd
    Y_j_prime=Y_j/Z_j_sd
    for int_loop in range(0,30):
        # (v2.5 insert C_eigen_val_count as the PCs truncation threshold)
        distances[int_loop,loc]=np.linalg.norm(X_prime[int_loop, 0:C_eigen_val_count]-Y_j_prime[0:C_eigen_val_count])
        print(loc,int_loop)
df=pd.DataFrame(distances)
df.to_excel(excel_writer = "analogs_temp.xlsx")
