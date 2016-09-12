from __future__ import print_function, division

import os, time
import contextlib, dbm
import random
import warnings
import itertools
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
import numpy as np


#Golobal configuration constant
train_data = './data/train.csv'
clean_train_data = './data/train.clean.csv'
shuffled_train_data = './data/train.clean.shuffled.csv'
valid_data = './data/validation.csv'
test_data = './data/test.csv' #all purchase of 2015

subsample_rate = 0.2 # sensible
batch_size = 5000;


#random.seed(1)

property_types = {'D': 0, 'S': 1, 'T': 2, 'F': 3, 'O': 4}
duration = {'F':0, 'L':1, 'U': 2}
city_features = None
cities_file = './data/cities.sorted.txt'

#---------------------------------
#   Visual and analyze data
#---------------------------------
#SKIP

#---------------------------------
#    Data preparation
#---------------------------------
def clean_data(original_file, clean_file):
    '''Simplifying, skip these steps'''
    pass 

def shuffle_data(train_file, shuffled_file):
    if os.path.isfile(shuffled_file):
        warnings.warn('Using the existed file: %s'%shuffled_file)
        return
    with contextlib.closing(dbm.open('temp', 'n')) as db:
        with open(train_file) as f:
            for i, line in enumerate(f):
                db[str(i)] = line
        linecount = i
        id_shuffle = list(range(linecount))
        random.shuffle(id_shuffle)
        with open(shuffled_file, 'w') as f:
            for i in id_shuffle:
                f.write(db[str(i)].decode('utf-8'))
    os.remove('temp.db')


def prepare_data(): 
    clean_data(train_data, clean_train_data)
    shuffle_data(clean_train_data, shuffled_train_data)
    #may consider here: scaling, decomposition, enrich features etc.

#---------------------------------
#    Feature extraction and baches divider
#---------------------------------
def subsampling(shuffled_file):
    '''For a fast development of a prototype.'''
    with open(shuffled_file) as f:
        for line in f:
            if random.random()<subsample_rate:
                yield line.strip()

def encode_one_hot(d, value):
    f = [0]*len(d.keys())
    f[d[value]] = 1
    return f

def build_binary_features(values_file):
    d = {}
    with open(values_file) as f:
        lines = f.readlines()
        max_value = len(lines)
        max_bin = format(max_value, '#b')
        max_bin = len(max_bin)
        for i, line in enumerate(lines):
            sb = format(i+1, '#0%db'%max_bin)[2:]
            d[line.strip()] = [int(c) for c in sb]

    return d

def encode_city_binary(value):
    global city_features
    if city_features is None:
        city_features = build_binary_features(cities_file)
    return city_features[value]

def get_feature_for_a_line(line):
    ms = line.split('","')
    y = float(ms[1])
    x = [(int(ms[2][:4])-1995)/(2016-1995)]#year of purchase, scaling to [0..1
    x.extend(encode_one_hot(property_types, ms[4]))#type of property
    x.extend([1] if ms[5] =='Y' else [0])#property status: old/new
    x.extend(encode_one_hot(duration, ms[6]))# lease duration
    x.extend(encode_city_binary(ms[11]))# city
    #TODO enrich features
    return x, y

def extract_features(rows):
    feature_size = 21
    X = np.zeros(shape=(len(rows), feature_size))#numpy fast allocate momory
    Y = np.zeros(shape=(len(rows),))
    for i, line in enumerate(rows):
        X[i], Y[i] = get_feature_for_a_line(line)
    return X, Y

def get_mini_batches_iter(shuffled_file, batch_size):
    row_iter = subsampling(shuffled_file)
    rows = [row for row in itertools.islice(row_iter, batch_size)]
    while(len(rows)!=0):
        X, Y = extract_features(rows)
        yield X, Y
        rows = [row for row in itertools.islice(row_iter, batch_size)]

#---------------------------------
#   Fit models/learning
#---------------------------------
def fit_partially(models, batches_iter):
    for i, (X, Y) in enumerate(batches_iter):
        for model in models:
            model.partial_fit(X, Y)
    return models

#---------------------------------
#    Evaluation
#---------------------------------
def evaluate(models, data):
    with open(data) as f:
        rows = f.readlines()
        X, Y = extract_features(rows)
        for model in models:
            print(type(model).__name__ + ' variance score:', model.score(X, Y))
            y_estimate = model.predict(X)
            print(type(model).__name__ + ' residual sum of squres:', np.mean((y_estimate-Y)**2))

 
#---------------------------------
#    Diagnotics, improvement, select/filter significal features (e.g Anova)
#---------------------------------


#---------------------------------
#    MAIN
#---------------------------------
def main():
    start = time.time()
    prepare_data()
    models = [SGDRegressor(), PassiveAggressiveRegressor()]
    
    num_iteration = 100
    for i in range(num_iteration):
        print('======Iteration #', i)
        mini_batches_iter = get_mini_batches_iter(shuffled_train_data, batch_size)
        fit_partially(models, mini_batches_iter)
        print('---Evaluation in validation set:')
        evaluate(models, valid_data)
        print('---Evaluation in test set:')
        evaluate(models, test_data)
        #TODO break if models don't learn anytihng new

if __name__ == '__main__':
    main()
