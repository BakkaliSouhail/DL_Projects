# Boltzmann Machines

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int64')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int64')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

# Converting the data into an array with users in lines and movies in colunms
def convert(data):
    new_data = []
    for id_users in range(1, 944):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings (Liked) or 0 (Not Liked)
training_set[training_set == 0] == -1
training_set[training_set == 1] == 0
training_set[training_set == 2] == 0
training_set[training_set >= 3] == 1
test_set[test_set == 0] == -1
test_set[test_set == 1] == 0
test_set[test_set == 2] == 0
test_set[test_set >= 3] == 1

# Creating the architecture of the Neural Network
class RBM():
    def __init__(self, NvisibleN, NhiddenN):
        self.Weights = torch.randn(NvisibleN, NhiddenN)
        self.Hbiases = torch.randn(1, NhiddenN)
        self.Vbiases = torch.randn(1, NvisibleN)

     def hidden_sample(self, x): #xcorresponds to the visible neurons where the coefficients are the weights
        Wx = torch.mm(x, self.Weights.t())
        # The activation function is a probability, it will be the probability that the hidden node  will be activated, according to the value of the visible node
        Activation_function = Wx + self.Hbiases.expand_as(Wx) #the bias is applied to each line of the minibatch
        # the probability that the hidden node iis activated, given the value of the visible node
        ProbHGV = torch.sigmoid(Activation_function)
        return PribHGV, torch.bernoulli(PribHGV) 
        
    def visible_sample(self, y):
        Wy = torch.mm(x, self.Weights)
        # The activation function is a probability, it will be the probability that the hidden node  will be activated, according to the value of the visible node
        Activation_function = Wy + self.Hbiases.expand_as(Wy) #the bias is applied to each line of the minibatch
        # the probability that the hidden node iis activated, given the value of the visible node
        ProbVGH = torch.sigmoid(Activation_function)
        return PribVGH, torch.bernoulli(PribVGH) 
        
    
        
        
        
        
        


 