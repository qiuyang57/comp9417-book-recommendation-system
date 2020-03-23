#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 17:33:49 2018

@author: root
"""

import pandas as pd
import datetime
import csv 
import numpy as np
import re
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import pairwise_distances
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from math import sqrt
#/////////////Processing data/////////////////////////

def processing_books():
    Books = {'ISBN':[],'BookTitle':[],'BookAuthor':[],'YearOfPublication':[],'Publisher':[]}
    with open('BX-Books.csv',encoding='latin1',newline='') as file:
        reader = csv.reader(file)
        count=0
        z=0
        for raw_data in reader:
            if count==0:
                count+=1
                continue
            temp=''.join(i for i in raw_data)
            temp=temp.replace('&amp;','')
            temp=temp.replace('"','')
            temp=temp.replace("\\",'')
            data=temp.split(";")
    #         print(data)
    #         if data[0]=='080213081X':
    #             print(raw_data)
            z+=1
            for i in range(1,len(data)):
                if data[i].isdigit():
                    if(i-1 >1):
                        data[1:i-1]=[''.join(data[1:i-1])]
                    if 4<len(data)-3:
                        data[4:len(data)-3]=[''.join(data[4:len(data)-3])]
                    break
            if len(data)>=5:
                Books['ISBN'].append(data[0])
                Books['BookTitle'].append(data[1])
                Books['BookAuthor'].append(data[2].replace("/"," "))
                if data[3].isdigit():
                    Books['YearOfPublication'].append(data[3])
                    Books['Publisher'].append(data[4])
                elif data[4].isdigit():
                    Books['YearOfPublication'].append(data[4])
                    Books['Publisher'].append(data[3])
                else:
                    pass
    #                 print(z,data)
            else:
                pass
                if len(data)>=5:
                    print(data)
    # for key in Books:
    #     print(len(Books[key]))
    file.close()
    books=pd.DataFrame(data=Books)
    books.YearOfPublication=pd.to_numeric(books.YearOfPublication,errors='coerce')
    now = datetime.datetime.now()
    books.loc[(books.YearOfPublication>now.year),'YearOfPublication']=now.year
    books.loc[(books.YearOfPublication==0),'YearOfPublication']=np.NAN
    year_mean=round(books.YearOfPublication.mean())
    books.YearOfPublication.fillna(year_mean,inplace=True)
    books.YearOfPublication = books.YearOfPublication.astype(np.int32)
    return books

def processing_users():
    title = 0
    User_data = {'User_ID':[],'City':[],'State':[],'Country':[],'Age':[]}
    with open('BX-Users.csv',encoding = 'latin1',newline='') as file:
        reader = csv.reader(file, delimiter=';', quotechar='"')
        for data in reader:
            if title == 0:
                title += 1
                continue
            location = data[1].split(",")
            location[-1] = re.sub(r'\.','',location[-1])
            if len(location) == 1:
                location = ['','','']
            if len(location) == 2:
                location.append('usa')
            User_data['User_ID'].append(data[0])
            User_data['City'].append(location[0])
            User_data['State'].append(location[1])
            User_data['Country'].append(location[2])
            User_data['Age'].append(data[2])

    file.close()
    df = pd.DataFrame(data = User_data)
    df.Age=pd.to_numeric(df.Age,errors='coerce')
    df.loc[(df.Age>116),'Age']=116
    df.loc[(df.Age == 0 ),'Age'] = np.NAN
    df.loc[(df.Age < 3 ),'Age'] = np.NAN
    age_mean=round(df.Age.mean())
    df.Age.fillna(age_mean,inplace=True)
    df.Age = df.Age.astype(np.int32)
    return df
def processing_rating():
    title = 0
    User_Book_Rating = {'User_ID':[],'ISBN':[],'Rating':[]}
    with open('BX-Book-Ratings.csv',encoding = 'latin1',newline='') as file:
        reader = csv.reader(file, delimiter=';', quotechar='"')
        for data in reader:
            if title == 0:
                title += 1
                continue
            User_Book_Rating['User_ID'].append(data[0])
            User_Book_Rating['ISBN'].append(data[1])
            User_Book_Rating['Rating'].append(data[2])
    file.close()
    df = pd.DataFrame(data = User_Book_Rating)
    df.Rating = df.Rating.astype(np.int32)
    df.User_ID = df.User_ID.astype(np.int32)
    return df

#/////////////Cosine similarity ,User base////////////////////
    
def predict_based_on_all_users (user_sim, training_set, testing_set):
    prediction_matrix=np.zeros(testing_set.shape)
    for user in range(len(prediction_matrix)):
        user_index=user_sim.index.get_loc(testing_set.index[user])
        user_id = np.array([np.argsort(user_sim.values[:,user_index])][0][::-1])
    #        for item in range(training_set.shape[1]):
        for item in testing_set.values[user,:].nonzero()[0]:
            temp = user_id[[training_set.values[:,item][user_id]>0]]
            if(temp.shape[0]==0):
                prediction_matrix[user, item] = 0
            else:
                top_k_user_id=temp
                denominator = np.sum(user_sim.values[user_index,:][top_k_user_id])
                numerator = user_sim.values[user_index,:][top_k_user_id].dot(training_set.values[:,item][top_k_user_id])
                if denominator==0:
                    prediction_matrix[user, item] = 0
                else:
                    prediction_matrix[user, item] = int(numerator/denominator)
    true_values = testing_set.values[testing_set.values.nonzero()].flatten()
    predicted_values = prediction_matrix[testing_set.values.nonzero()].flatten()
    mse = mean_squared_error(predicted_values, true_values)
    RMSE = round(sqrt(mse),3)
    print('The RMSE of all  user_based CF is: ' + str(RMSE) + '\n')
    return RMSE

#def predict_based_on_topk_users (k, user_sim, training_set, testing_set):
#    prediction_matrix=np.zeros(testing_set.shape)
#    for user in range(len(prediction_matrix)):
#        user_index=user_sim.index.get_loc(testing_set.index[user])
#        user_id = [np.argsort(user_sim.values[:,user_index])]
#        for item in range(training_set.shape[1]):
#            top_k_user_id=[]
#            for i in user_id[0][::-1]:
#                if(training_set.values[i,item]!=0):
#                    top_k_user_id.append(i)
#                if len(top_k_user_id)==k:
#                    break
#            top_k_user_id=np.array(top_k_user_id)
#            denominator = np.sum(user_sim.values[user_index,:][top_k_user_id])
#            numerator = user_sim.values[user_index,:][top_k_user_id].dot(training_set.values[:,item][top_k_user_id])
#            prediction_matrix[user, item] = int(numerator/denominator)
#    true_values = testing_set.values[testing_set.values.nonzero()].flatten()
#    predicted_values = prediction_matrix[testing_set.values.nonzero()].flatten()
#    mse = mean_squared_error(predicted_values, true_values)
#    print('The mean squared error of top-' + str(k) + ' user_based CF is: ' + str(mse) + '\n')
#    return mse,prediction_matrix

def predict_based_on_topk_users_a (k, user_sim, training_set, testing_set):
    prediction_matrix=np.zeros(testing_set.shape)
    for user in range(len(prediction_matrix)):
        user_index=user_sim.index.get_loc(testing_set.index[user])
        user_id = np.array([np.argsort(user_sim.values[:,user_index])][0][::-1])
#        for item in range(training_set.shape[1]):
        for item in testing_set.values[user,:].nonzero()[0]:
            temp = user_id[[training_set.values[:,item][user_id]>0]]
            if(temp.shape[0]==0):
                prediction_matrix[user, item] = 0
            else:
                if(temp.shape[0]>k):
                    top_k_user_id=temp[:k]
                else:
                    top_k_user_id=temp
                denominator = np.sum(user_sim.values[user_index,:][top_k_user_id])
                numerator = user_sim.values[user_index,:][top_k_user_id].dot(training_set.values[:,item][top_k_user_id])
                if denominator==0:
                    prediction_matrix[user, item] = 0
                else:
                    prediction_matrix[user, item] = round(numerator/denominator)
    true_values = testing_set.values[testing_set.values.nonzero()].flatten()
    predicted_values = prediction_matrix[testing_set.values.nonzero()].flatten()
    mse = mean_squared_error(predicted_values, true_values)
    RMSE = round(sqrt(mse),3)
    print('The RMSE of top-' + str(k) + ' user_based CF is: ' + str(RMSE) + '\n')
#    return predicted_values,true_values
    return RMSE

def similarity(Rating_matrix):
    # sim[m ,n] = rating[m, :] X rating[n, :]
    # which is sum of movie ratings from each user u and different user u'
    # add 1e-9 make it non zero
    sim = np.dot(Rating_matrix, Rating_matrix.T) + 1e-9

    # the diagonal is just sqrt of user rating
    norms = np.array([np.sqrt(np.diagonal(sim))])
    
    return (sim / (norms * norms.T))

# ///////////////Pearson correlation User base//////////////////////////
    
def predict_based_on_topk_users_p (k, user_sim, training_set, testing_set,users_mean_rating):
    prediction_matrix=np.zeros(testing_set.shape)
    for user in range(len(prediction_matrix)):
        user_index=user_sim.index.get_loc(testing_set.index[user])
        user_id = np.array([np.argsort(user_sim.values[:,user_index])][0][::-1])
    #        for item in range(training_set.shape[1]):
        for item in testing_set.values[user,:].nonzero()[0]:
            temp = user_id[[training_set.values[:,item][user_id]>0]]
            if(temp.shape[0]==0):
                prediction_matrix[user, item] = users_mean_rating[user]
            else:
                if(temp.shape[0]>k):
                    top_k_user_id=temp[:k]
                else:
                    top_k_user_id=temp
                denominator = np.sum(user_sim.values[user_index,:][top_k_user_id])
                numerator = user_sim.values[user_index,:][top_k_user_id].dot(training_set.values[:,item][top_k_user_id]-users_mean_rating[top_k_user_id])
                if denominator==0:
                    prediction_matrix[user, item] = users_mean_rating[user]
                else:
                    prediction_matrix[user, item] = round(users_mean_rating[user]+ numerator/denominator)
    true_values = testing_set.values[testing_set.values.nonzero()].flatten()
    predicted_values = prediction_matrix[testing_set.values.nonzero()].flatten()
    mse = mean_squared_error(predicted_values, true_values)
    RMSE = round(sqrt(mse),3)
#    print('The RMSE of top-' + str(k) + ' user_based CF is: ' + str(RMSE) + '\n')
    print((k,RMSE),end=' ')
    return RMSE,predicted_values,true_values
#    return RMSE

def predict_based_on_topk_users_k (k, user_sim, training_set, testing_set,users_mean_rating):
    prediction_matrix=np.zeros(testing_set.shape)
    for user in range(len(prediction_matrix)):
        user_index=user_sim.index.get_loc(testing_set.index[user])
        user_id = np.array([np.argsort(user_sim.values[:,user_index])][0][::-1])
    #        for item in range(training_set.shape[1]):
        for item in testing_set.values[user,:].nonzero()[0]:
            temp = user_id[[training_set.values[:,item][user_id]>0]]
            if(temp.shape[0]==0):
                prediction_matrix[user, item] = users_mean_rating[user]
            else:
                if(temp.shape[0]>k):
                    top_k_user_id=temp[:k]
                else:
                    top_k_user_id=temp
                denominator = np.sum(user_sim.values[user_index,:][top_k_user_id])
                numerator = user_sim.values[user_index,:][top_k_user_id].dot(training_set.values[:,item][top_k_user_id]-users_mean_rating[top_k_user_id])
                if denominator==0:
                    prediction_matrix[user, item] = users_mean_rating[user]
                else:
                    prediction_matrix[user, item] = round(users_mean_rating[user]+ numerator/denominator)
    true_values = testing_set.values[testing_set.values.nonzero()].flatten()
    predicted_values = prediction_matrix[testing_set.values.nonzero()].flatten()
    mse = mean_squared_error(predicted_values, true_values)
    RMSE = round(sqrt(mse),3)
    print('The RMSE of top-' + str(k) + ' user_based CF is: ' + str(RMSE) + '\n')
    return predicted_values,true_values
#    return RMSE

def SVD_model_fit_and_predict(k,training_matrix,testing_matrix):
    training_matrix_nan=training_matrix.copy()
    training_matrix_nan[training_matrix_nan==0]=np.nan
    tm = training_matrix_nan.values
    tm_mean=np.nanmean(tm,axis=0,keepdims=True)
    tm=tm-tm_mean
    tm[np.isnan(tm)]=0
    ts = csc_matrix(tm).asfptype()
    
    u, s, vt = svds(ts, k = i)
    s_diag_matrix=np.diag(s)
    X_pred = np.around(np.dot(np.dot(u, s_diag_matrix), vt)+tm_mean)
    nz=testing_matrix.values.nonzero()
    tv = testing_matrix.values[nz[0],nz[1]]
    pv = X_pred[nz[0],nz[1]]
    mse=((pv-tv) ** 2).mean(axis=0)
    nnz=nnan_rating_matrix.values.nonzero()
    ttv = training_matrix.values[nnz]
    ppv=X_pred[nnz[0],nnz[1]]
    mmse=((ppv-ttv) ** 2).mean(axis=0)
    train_rmse = round(sqrt(mmse),3)
    RMSE = round(sqrt(mse),3)
    print(train_rmse,RMSE)
    rmse.append(RMSE)
    return train_rmse,rmse,X_pred

#/////////////////main part//////////////////
#%%
books=processing_books()

users=processing_users()

rating=processing_rating()
#%%
new_rating=rating[rating.ISBN.isin(books.ISBN)]

new_rating=new_rating[new_rating.User_ID.isin(users.User_ID)]

rating_exp = new_rating[new_rating.Rating!=0]
#%%
# build user-item matrix, only consider  users who have rated at least 100 books and books which have at least 100 ratings. 

previous_shape=np.inf
while (rating_exp.shape[0]<previous_shape):
    previous_shape = rating_exp.shape[0]
    counts=rating_exp['ISBN'].value_counts()
    
    rating_exp=rating_exp[rating_exp['ISBN'].isin(counts[counts>=7].index)]
    
    counts1=rating_exp['User_ID'].value_counts()
    
    rating_exp=rating_exp[rating_exp['User_ID'].isin(counts1[counts1>=7].index)]

#%%
#rating_exp=rating_exp.reset_index(drop=True)
rating_exp=rating_exp.sample(frac=1).reset_index(drop=True)
rating_matrix=rating_exp.pivot(index='User_ID',columns='ISBN',values='Rating')
rating_matrix_item = rating_exp.pivot(index='ISBN',columns='User_ID',values='Rating')
#%%
#
#matrix_size = rating_matrix.shape[0]*rating_matrix.shape[1]
#
#matrix_sparsity = float(rating_exp.shape[0])/matrix_size
#
#rating_matrix=rating_matrix.fillna(0)
#
#rating_matrix = rating_matrix.astype(np.int32)

#split data to test data and traing data

#test_data_size = math.floor(0.1 * matrix_sparsity *rating_matrix.shape[1])

#n_users = rating_matrix.shape[0]

#n_items = rating_matrix.shape[1]

#testing_data_set = pd.DataFrame(np.zeros((n_users, n_items)),index=rating_matrix.index,columns=rating_matrix.columns)
#
#training_data_set = rating_matrix.copy()
#
#for uid in range(n_users):
#    
#    item = np.random.choice(rating_matrix.values[uid, :].nonzero()[0], size=test_data_size, replace=False)
#    
##    print(item)
#    
#    testing_data_set.values[uid, item] = rating_matrix.values[uid, item]
#    
#    training_data_set.values[uid, item] = 0.
    
# Pearson correlation
    
#user_sim = pd.DataFrame(data=1-pairwise_distances(training_data_set, metric="correlation"),index=rating_matrix.index,columns=rating_matrix.index)


#%%
kf = RepeatedKFold(n_splits=10,n_repeats=1)

nnan_rating_matrix = rating_matrix.fillna(0)
nnan_rating_matrix_item = rating_matrix_item.fillna(0)
#%%
pearson,euclidean,cosine,mean_centered_cos,mean_centered_cos1=[],[],[],[],[]
pearson_i,euclidean_i,cosine_i,mean_centered_cos_i,mean_centered_cos_i1=[],[],[],[],[]
for k in range(1,50,3):
    for train_index, test_index in kf.split(rating_exp):
        training_set=rating_exp.iloc[train_index]
        testing_set= rating_exp.iloc[test_index]
        n_users = rating_matrix.shape[0]
    
        n_items = rating_matrix.shape[1]
        
        testing_matrix = pd.DataFrame(np.zeros((n_users, n_items)),index=rating_matrix.index,columns=rating_matrix.columns)
        training_matrix=nnan_rating_matrix.copy()
        
        testing_matrix_item = pd.DataFrame(np.zeros((n_items,n_users)),index=rating_matrix_item.index,columns=rating_matrix_item.columns)
        training_matrix_item = nnan_rating_matrix_item.copy()
        
        user_sim = pd.DataFrame(data=1-pairwise_distances(training_matrix, metric="correlation"),index=rating_matrix.index,columns=rating_matrix.index)
    #    user_sim = pd.DataFrame(data=similarity(1-pairwise_distances(training_matrix, metric="correlation")),index=rating_matrix.index,columns=rating_matrix.index)
    
        user_sim_item = pd.DataFrame(data=1-pairwise_distances(training_matrix_item, metric="correlation"),index=rating_matrix_item.index,columns=rating_matrix_item.index)
    #    user_sim_item = pd.DataFrame(data=similarity(1-pairwise_distances(training_matrix_item, metric="correlation")),index=rating_matrix_item.index,columns=rating_matrix_item.index)
        
        user_sim1 = pd.DataFrame(data=1/(1+pairwise_distances(training_matrix, metric="euclidean")),index=rating_matrix.index,columns=rating_matrix.index)
    #    user_sim1 = pd.DataFrame(data=similarity(1/(1+pairwise_distances(training_matrix, metric="euclidean"))),index=rating_matrix.index,columns=rating_matrix.index)
    
        user_sim1_item = pd.DataFrame(data=1/(1+pairwise_distances(training_matrix_item, metric="euclidean")),index=rating_matrix_item.index,columns=rating_matrix_item.index)
    #    user_sim1_item = pd.DataFrame(data=similarity(1/(1+pairwise_distances(training_matrix_item, metric="euclidean"))),index=rating_matrix_item.index,columns=rating_matrix_item.index)
        
        user_sim2 = pd.DataFrame(data=cosine_similarity(training_matrix),index=rating_matrix.index,columns=rating_matrix.index)
    #    user_sim2 = pd.DataFrame(data=similarity(cosine_similarity(training_matrix)),index=rating_matrix.index,columns=rating_matrix.index)
        
        user_sim2_item = pd.DataFrame(data=cosine_similarity(training_matrix_item),index=rating_matrix_item.index,columns=rating_matrix_item.index)
    #    user_sim2_item = pd.DataFrame(data=similarity(cosine_similarity(training_matrix_item)),index=rating_matrix_item.index,columns=rating_matrix_item.index)
    

    
    
        temp_rating=training_matrix.values.astype(float)
        temp_rating_item=training_matrix_item.values.astype(float)
    
        temp_rating[temp_rating==0]=np.nan
        temp_rating_item[temp_rating_item==0]=np.nan
        
        users_mean_rating = np.nanmean(temp_rating, axis=1)
        users_mean_rating_item = np.nanmean(temp_rating_item, axis=1)
        
        for index,row in testing_set.iterrows():
            training_matrix.loc[row['User_ID'],row['ISBN']]=0
            testing_matrix.loc[row['User_ID'],row['ISBN']]=nnan_rating_matrix.loc[row['User_ID'],row['ISBN']]
    
            training_matrix_item.loc[row['ISBN'],row['User_ID']]=0
            testing_matrix_item.loc[row['ISBN'],row['User_ID']]=nnan_rating_matrix_item.loc[row['ISBN'],row['User_ID']]
    
        pearson.append(predict_based_on_topk_users_p(k, user_sim,training_matrix,testing_matrix,users_mean_rating))
        pearson_i.append(predict_based_on_topk_users_p(k, user_sim_item,training_matrix_item,testing_matrix_item,users_mean_rating_item))
        euclidean.append(predict_based_on_topk_users_p(k, user_sim1,training_matrix,testing_matrix,users_mean_rating))
        euclidean_i.append(predict_based_on_topk_users_p(k, user_sim1_item,training_matrix_item,testing_matrix_item,users_mean_rating_item))
        cosine.append(predict_based_on_topk_users_p(k, user_sim2,training_matrix,testing_matrix,users_mean_rating))
        cosine_i.append(predict_based_on_topk_users_p(k, user_sim2_item,training_matrix_item,testing_matrix_item,users_mean_rating_item))
    
    p = np.mean([x[0] for x in pearson])
    p_i = np.mean([x[0] for x in pearson_i])
    e = np.mean([x[0] for x in euclidean])
    e_i = np.mean([x[0] for x in euclidean_i])
    c = np.mean([x[0] for x in cosine])
    c_i = np.mean([x[0] for x in cosine_i])

    r=[k,p,p_i,e,e_i,c,c_i]
#%%
labels=['pearson_user_base','person_item_base','euclidean_user_base','euclidean_item_base','cosine_user_base','cosine_item_base']
colors=['C0','C1','C2','C3','C4','C5',]


#%%    
print(np.mean([x[0] for x in pearson]))
print(np.mean([x[0] for x in euclidean]))
print(np.mean([x[0] for x in cosine]))
print(np.mean([x[0] for x in pearson_i]))
print(np.mean([x[0] for x in euclidean_i]))
print(np.mean([x[0] for x in cosine_i]))
#%%
print(np.min([x[0] for x in pearson]))
print(np.min([x[0] for x in euclidean]))
print(np.min([x[0] for x in cosine]))
print(np.min([x[0] for x in pearson_i]))
print(np.min([x[0] for x in euclidean_i]))
print(np.min([x[0] for x in cosine_i]))

#%%
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



#%%
%matplotlib inline
distri_df = pd.DataFrame(ttv)
distri_df.hist(bins=np.arange(start=1, stop=10, step=1))

#%%    
count=0
for i in range(pv.shape[0]):
    if pv[i]==tv[i]:
        count+=1
print(count)    

#%%
#Cosine similarity

#user_sim = pd.DataFrame(data=cosine_similarity(training_data_set),index=rating_matrix.index,columns=rating_matrix.index)

##### Test############

#performance=[]
#
#k_list = [i for i in range(50,training_data_set.shape[0],50)]
#
#for k in k_list:
#    
#    performance.append(predict_based_on_topk_users_a (k, user_sim, training_data_set, testing_data_set))
#    
#performance.append(predict_based_on_all_users(user_sim,training_data_set,testing_data_set))