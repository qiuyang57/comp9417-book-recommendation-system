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
# from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
# from scipy.spatial.distance import correlation, cosine
# import ipywidgets as widgets
# from IPython.display import display, clear_output
# from sklearn.metrics import pairwise_distances
from sklearn.metrics import mean_squared_error


# from math import sqrt
# import matplotlib.pyplot as plt
# import sklearn.metrics as metrics
def processing_books():
    Books = {'ISBN': [], 'BookTitle': [], 'BookAuthor': [], 'YearOfPublication': [], 'Publisher': []}
    with open('BX-Books.csv', encoding='latin1', newline='') as file:
        reader = csv.reader(file)
        count = 0
        z = 0
        for raw_data in reader:
            if count == 0:
                count += 1
                continue
            temp = ''.join(i for i in raw_data)
            temp = temp.replace('&amp;', '')
            temp = temp.replace('"', '')
            temp = temp.replace("\\", '')
            data = temp.split(";")
            #         print(data)
            #         if data[0]=='080213081X':
            #             print(raw_data)
            z += 1
            for i in range(1, len(data)):
                if data[i].isdigit():
                    if (i - 1 > 1):
                        data[1:i - 1] = [''.join(data[1:i - 1])]
                    if 4 < len(data) - 3:
                        data[4:len(data) - 3] = [''.join(data[4:len(data) - 3])]
                    break
            if len(data) >= 5:
                Books['ISBN'].append(data[0])
                Books['BookTitle'].append(data[1])
                Books['BookAuthor'].append(data[2].replace("/", " "))
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
                if len(data) >= 5:
                    print(data)
    # for key in Books:
    #     print(len(Books[key]))
    file.close()
    books = pd.DataFrame(data=Books)
    books.YearOfPublication = pd.to_numeric(books.YearOfPublication, errors='coerce')
    now = datetime.datetime.now()
    books.loc[(books.YearOfPublication > now.year), 'YearOfPublication'] = now.year
    books.loc[(books.YearOfPublication == 0), 'YearOfPublication'] = np.NAN
    year_mean = round(books.YearOfPublication.mean())
    books.YearOfPublication.fillna(year_mean, inplace=True)
    books.YearOfPublication = books.YearOfPublication.astype(np.int32)
    return books


def processing_users():
    title = 0
    User_data = {'User_ID': [], 'City': [], 'State': [], 'Country': [], 'Age': []}
    with open('BX-Users.csv', encoding='latin1', newline='') as file:
        reader = csv.reader(file, delimiter=';', quotechar='"')
        for data in reader:
            if title == 0:
                title += 1
                continue
            location = data[1].split(",")
            location[-1] = re.sub(r'\.', '', location[-1])
            if len(location) == 1:
                location = ['', '', '']
            if len(location) == 2:
                location.append('usa')
            User_data['User_ID'].append(data[0])
            User_data['City'].append(location[0])
            User_data['State'].append(location[1])
            User_data['Country'].append(location[2])
            User_data['Age'].append(data[2])

    file.close()
    df = pd.DataFrame(data=User_data)
    df.Age = pd.to_numeric(df.Age, errors='coerce')
    df.loc[(df.Age > 116), 'Age'] = 116
    df.loc[(df.Age == 0), 'Age'] = np.NAN
    df.loc[(df.Age < 3), 'Age'] = np.NAN
    age_mean = round(df.Age.mean())
    df.Age.fillna(age_mean, inplace=True)
    df.Age = df.Age.astype(np.int32)
    return df


def processing_rating():
    title = 0
    User_Book_Rating = {'User_ID': [], 'ISBN': [], 'Rating': []}
    with open('BX-Book-Ratings.csv', encoding='latin1', newline='') as file:
        reader = csv.reader(file, delimiter=';', quotechar='"')
        for data in reader:
            if title == 0:
                title += 1
                continue
            User_Book_Rating['User_ID'].append(data[0])
            User_Book_Rating['ISBN'].append(data[1])
            User_Book_Rating['Rating'].append(data[2])
    file.close()
    df = pd.DataFrame(data=User_Book_Rating)
    df.Rating = df.Rating.astype(np.int32)
    df.User_ID = df.User_ID.astype(np.int32)
    return df


def predict_based_on_all_users(user_sim, training_set, testing_set):
    prediction_matrix = np.zeros(testing_set.shape)
    for user in range(len(prediction_matrix)):
        user_index = user_sim.index.get_loc(testing_set.index[user])
        for item in range(training_set.shape[1]):
            denominator = np.sum(user_sim.values[user_index, :])
            numerator = user_sim.values[user_index, :].dot(training_set.values[:, item])
            prediction_matrix[user, item] = int(numerator / denominator)
    true_values = testing_set.values[testing_set.values.nonzero()].flatten()
    predicted_values = prediction_matrix[testing_set.values.nonzero()].flatten()
    mse = mean_squared_error(predicted_values, true_values)
    print('The mean squared error of user_based CF is: ' + str(mse) + '\n')
    return mse, prediction_matrix


def predict_based_on_topk_users(k, user_sim, training_set, testing_set):
    prediction_matrix = np.zeros(testing_set.shape)
    for user in range(len(prediction_matrix)):
        user_index = user_sim.index.get_loc(testing_set.index[user])
        top_k_user_id = [np.argsort(user_sim.values[:, user_index])[-2:-k - 2:-1]]
        for item in range(training_set.shape[1]):
            denominator = np.sum(user_sim.values[user_index, :][top_k_user_id])
            numerator = user_sim.values[user_index, :][top_k_user_id].dot(training_set.values[:, item][top_k_user_id])
            prediction_matrix[user, item] = int(numerator / denominator)
    true_values = testing_set.values[testing_set.values.nonzero()].flatten()
    predicted_values = prediction_matrix[testing_set.values.nonzero()].flatten()
    mse = mean_squared_error(predicted_values, true_values)
    print('The mean squared error of top-' + str(k) + ' user_based CF is: ' + str(mse) + '\n')
    return mse, prediction_matrix


# def similarity(Rating_matrix):
#    # sim[m ,n] = rating[m, :] X rating[n, :]
#    # which is sum of movie ratings from each user u and different user u'
#    # add 1e-9 make it non zero
#    sim = np.dot(Rating_matrix, Rating_matrix.T) + 1e-9
#
#    # the diagonal is just sqrt of user rating
#    norms = np.array([np.sqrt(np.diagonal(sim))])
#    
#    return (sim / (norms * norms.T))
# def mean_squared_error(y_true, y_pred):
#    
#    return np.average((y_true - y_pred) ** 2)

books = processing_books()
users = processing_users()
rating = processing_rating()
new_rating = rating[rating.ISBN.isin(books.ISBN)]
new_rating = new_rating[new_rating.User_ID.isin(users.User_ID)]
rating_exp = new_rating[new_rating.Rating != 0]
# build user-item matrix, only consider  users who have rated at least 100 books and books which have at least 100 ratings. 
counts1 = rating_exp['User_ID'].value_counts()
rating_exp = rating_exp[rating_exp['User_ID'].isin(counts1[counts1 >= 100].index)]
counts = rating_exp['ISBN'].value_counts()
rating_exp = rating_exp[rating_exp['ISBN'].isin(counts[counts >= 10].index)]
rating_matrix = rating_exp.pivot(index='User_ID', columns='ISBN', values='Rating')

matrix_size = rating_matrix.shape[0] * rating_matrix.shape[1]
matrix_sparsity = float(rating_exp.shape[0]) / matrix_size
rating_matrix = rating_matrix.fillna(0)
rating_matrix = rating_matrix.astype(np.int32)
# split data to test data and traing data
test_data_size = math.floor(0.1 * matrix_sparsity * rating_matrix.shape[1])
n_users = rating_matrix.shape[0]
n_items = rating_matrix.shape[1]
testing_data_set = pd.DataFrame(np.zeros((n_users, n_items)), index=rating_matrix.index, columns=rating_matrix.columns)
training_data_set = rating_matrix
for uid in range(n_users):
    item = np.random.choice(rating_matrix.values[uid, :].nonzero()[0], size=test_data_size, replace=False)
    #    print(item)
    testing_data_set.values[uid, item] = rating_matrix.values[uid, item]
    training_data_set.values[uid, item] = 0.
user_sim = pd.DataFrame(data=cosine_similarity(training_data_set), index=rating_matrix.index,
                        columns=rating_matrix.index)
