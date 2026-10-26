import pandas as pd
import csv
from requests import get
import json
from datetime import datetime, timedelta, date
import numpy as np
from scipy.spatial.distance import euclidean, cityblock, cosine
from scipy.stats import pearsonr

import csv
import re
import pandas as pd
import argparse
import collections
import json
import glob
import math
import os
import requests
import string
import sys
import time
import xml
import random

class Recommender(object):
    def __init__(self, training_set, test_set):
        if isinstance(training_set, str):
            # the training set is a file name
            self.training_set = pd.read_csv(training_set)
        else:
            # the training set is a DataFrame
            self.training_set = training_set.copy()

        if isinstance(test_set, str):
            # the test set is a file name
            self.test_set = pd.read_csv(test_set)
        else:
            # the test set is a DataFrame
            self.test_set = test_set.copy()
    
    def train_user_euclidean(self, data_set, userId):
        weights = {}
        target = data_set[userId]

        for other in data_set.columns:
            if other == userId or other == 'movieId':
                continue

            other_series = data_set[other]
            mask = target.notna() & other_series.notna()
            a = target[mask].values
            b = other_series[mask].values

            if len(a) == 0:
                continue

            weights[other] = 1.0 / (1.0 + euclidean(a, b))
        return weights # dictionary of weights mapped to users. e.g. {"0331949b45":1.0, "1030c5a8a9":2.5}
    
    def train_user_manhattan(self, data_set, userId):
        weights = {}
        target = data_set[userId]

        for other in data_set.columns:
            if other == userId or other == 'movieId':
                continue

            other_series = data_set[other]
            mask = target.notna() & other_series.notna()
            a = target[mask].values
            b = other_series[mask].values

            if len(a) == 0:
                continue

            weights[other] = 1.0 / (1.0 + cityblock(a, b))
        return weights # dictionary of weights mapped to users. e.g. {"0331949b45":1.0, "1030c5a8a9":2.5}

    def train_user_cosine(self, data_set, userId):
        weights = {}
        target = data_set[userId]

        for other in data_set.columns:
            if other == userId or other == 'movieId':
                continue

            other_series = data_set[other]
            mask = target.notna() & other_series.notna()
            a = target[mask].values
            b = other_series[mask].values

            if len(a) == 0:
                continue
            if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
                continue

            weights[other] = 1.0 - cosine(a, b)

        return weights # dictionary of weights mapped to users. e.g. {"0331949b45":1.0, "1030c5a8a9":2.5}
   
    def train_user_pearson(self, data_set, userId):
        weights = {}
        target = data_set[userId]

        for other in data_set.columns:
            if other == userId or other == 'movieId':
                continue

            other_series = data_set[other]
            mask = target.notna() & other_series.notna()
            a = target[mask].values
            b = other_series[mask].values

            if len(a) < 2:
                continue

            r = pearsonr(a, b)[0]
            
            weights[other] = r

        return weights # dictionary of weights mapped to users. e.g. {"0331949b45":1.0, "1030c5a8a9":2.5}

    def train_user(self, data_set, distance_function, userId):
        if distance_function == 'euclidean':
            return self.train_user_euclidean(data_set, userId)
        elif distance_function == 'manhattan':
            return self.train_user_manhattan(data_set, userId)
        elif distance_function == 'cosine':
            return self.train_user_cosine(data_set, userId)
        elif distance_function == 'pearson':
            return self.train_user_pearson(data_set, userId)
        else:
            return None

    def get_user_existing_ratings(self, data_set, userId):
        series = data_set.set_index('movieId')[userId].dropna()
        return [(int(m), float(r)) for m, r in series.items()] # list of tuples with movieId and rating. e.g. [(32, 4.0), (50, 4.0)]

    def predict_user_existing_ratings_top_k(self, data_set, sim_weights, userId, k):
        result = []
        indexed = data_set.set_index('movieId')
        user_ratings = indexed[userId].dropna()
        sorted_weights = sorted(sim_weights.items(), key=lambda x: x[1], reverse=True)
        top_k_users = sorted_weights[:k]

        for movieId in user_ratings.index:
            movie_row = indexed.loc[movieId]
            num = 0.0
            denom = 0.0

            for other, weight in top_k_users:
                r = movie_row[other]
                if pd.isna(r):
                    continue
                num += weight * r
                denom += (weight)

            if denom == 0:
                continue

            result.append((int(movieId), num / denom))
        return result # list of tuples with movieId and rating. e.g. [(32, 4.0), (50, 4.0)]
    
    def evaluate(self, existing_ratings, predicted_ratings):
        existing_dict = dict(existing_ratings)
        predicted_dict = dict(predicted_ratings)

        baseline_movies = []
        for m in existing_dict:
            if pd.isna(existing_dict[m]):
                continue
            baseline_movies.append(m)

        if not baseline_movies:
            return {'rmse': 0, 'ratio': 0}

        valid_common = []
        for m in baseline_movies:
            if m not in predicted_dict:
                continue
            if pd.isna(predicted_dict[m]):
                continue
            valid_common.append(m)

        ratio = len(valid_common) / len(baseline_movies)

        if not valid_common:
            return {'rmse': 0, 'ratio': ratio}

        sq_errors = [(existing_dict[m] - predicted_dict[m]) ** 2 for m in valid_common]
        rmse = math.sqrt(sum(sq_errors) / len(valid_common))

        return {'rmse': rmse, 'ratio': ratio} # dictionary with an rmse value and a ratio. e.g. {'rmse':1.2, 'ratio':0.5}
    
    def single_calculation(self, distance_function, userId, k_values):
        user_existing_ratings = self.get_user_existing_ratings(self.test_set, userId)
        print("User has {} existing and {} missing movie ratings".format(len(user_existing_ratings), len(self.test_set) - len(user_existing_ratings)), file=sys.stderr)

        print('Building weights')
        sim_weights = self.train_user(self.training_set[self.test_set.columns.values.tolist()], distance_function, userId)

        result = []
        for k in k_values:
            print('Calculating top-k user prediction with k={}'.format(k))
            top_k_existing_ratings_prediction = self.predict_user_existing_ratings_top_k(self.test_set, sim_weights, userId, k)
            result.append((k, self.evaluate(user_existing_ratings, top_k_existing_ratings_prediction)))
        return result # list of tuples, each of which has the k value and the result of the evaluation. e.g. [(1, {'rmse':1.2, 'ratio':0.5}), (2, {'rmse':1.0, 'ratio':0.9})]

    def aggregate_calculation(self, distance_functions, userId, k_values):
        print()
        result_per_k = {}
        for func in distance_functions:
            print("Calculating for {} distance metric".format(func))
            for calc in self.single_calculation(func, userId, k_values):
                if calc[0] not in result_per_k:
                    result_per_k[calc[0]] = {}
                result_per_k[calc[0]]['{}_rmse'.format(func)] = calc[1]['rmse']
                result_per_k[calc[0]]['{}_ratio'.format(func)] = calc[1]['ratio']
            print()
        result = []
        for k in k_values:
            row = {'k':k}
            row.update(result_per_k[k])
            result.append(row)
        columns = ['k']
        for func in distance_functions:
            columns.append('{}_rmse'.format(func))
            columns.append('{}_ratio'.format(func))
        result = pd.DataFrame(result, columns=columns)
        return result
        
if __name__ == "__main__":
    recommender = Recommender("data/train.csv", "data/medium_test.csv")
    print("Training set has {} users and {} movies".format(len(recommender.training_set.columns[1:]), len(recommender.training_set)))
    print("Testing set has {} users and {} movies".format(len(recommender.test_set.columns[1:]), len(recommender.test_set)))

    result = recommender.aggregate_calculation(['euclidean', 'cosine', 'pearson', 'manhattan'], "02839371ac", [1, 2, 3, 4])
    print(result)