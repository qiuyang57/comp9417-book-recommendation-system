# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 23:28:01 2018

@author: qiuya
"""

import scipy.sparse as sp
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix

rmse=[]
data=[]
for i in range(1,200,20):
    result=[]
    for train_index, test_index in kf.split(rating_exp):
        training_set=rating_exp.iloc[train_index]
        testing_set= rating_exp.iloc[test_index]
        n_users = rating_matrix.shape[0]
    
        n_items = rating_matrix.shape[1]
        
        testing_matrix = pd.DataFrame(np.zeros((n_users, n_items)),index=rating_matrix.index,columns=rating_matrix.columns)
        training_matrix=nnan_rating_matrix.copy()
        
          
        for index,row in testing_set.iterrows():
            training_matrix.loc[row['User_ID'],row['ISBN']]=0
            testing_matrix.loc[row['User_ID'],row['ISBN']]=nnan_rating_matrix.loc[row['User_ID'],row['ISBN']]
        
    
        result.append(SVD_model_fit_and_predict(i,training_matrix,testing_matrix))
    data.append((i,np.mean([x[0] for x in result]),np.mean([x[1] for x in result])))
#%%
plt.plot([x[0] for x in data],[x[1] for x in data],label='training_set_RMSE')
plt.plot([x[0] for x in data],[x[2] for x in data],label='testing_set_RMSE')
plt.ylabel('rmse')
plt.xlabel('Number of singular values')
plt.legend()
plt.show()