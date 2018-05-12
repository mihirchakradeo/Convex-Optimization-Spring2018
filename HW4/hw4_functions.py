######
######  This file includes different functions used in HW3
######

import numpy as np
import matplotlib.pyplot as plt
import time
import random
import math

def svm_objective_function(w, features, labels, order):
    n=len(labels)
    if order==0:
        # value = ( TODO: value )
        value = np.mean(np.maximum(1 - np.multiply(labels, features.dot(w)), 0))
        return value
    elif order==1:
        # value = ( TODO: value )
        # value = np.mean(np.maximum(1 - np.multiply(labels, features.dot(w)), 0))
        # print("value",value)
        value = np.inf

        # subgradient = ( TODO: sungradient )
        temp = np.maximum(1 - np.multiply(labels, features.dot(w)), 0)
        temp[temp>0] = 1
        temp_sd = -np.multiply(labels, features)
        subgradient = np.multiply(temp, temp_sd)
        subgradient = np.sum(subgradient, axis = 0)
        subgradient = (1/n)*subgradient

        return(value,subgradient.T)


    else:
        raise ValueError("The argument \"order\" should be 0 or 1")

def svm_objective_function_stochastic(w, features, labels, order, minibatch_size):
    # print("shape",features.shape, w.shape, labels.shape)
    n=len(labels)
    if order==0:
        # value = ( TODO: value )
        # value = np.mean(max(1 - labels*w.T.dot(features), 0))
        value = np.mean(np.maximum(1 - np.multiply(labels, features.dot(w)), 0))
        print("value0",value)

        return value
    elif order==1:
        # value = ( TODO: value )
        # value = np.mean(np.maximum(1 - np.multiply(labels, features.dot(w)), 0))
        # print("value1",value)
        value = np.inf
        # subgradient = ( TODO: sungradient )

        indices = np.random.choice(len(labels), minibatch_size, replace=False)

        labels = labels[indices]
        # print("labels.shape",labels.shape)

        features = features[indices]
        # print("features",features.shape)


        temp = np.maximum(1 - np.multiply(labels, features.dot(w)), 0)
        # print("np.multiply(labels, features.dot(w))", np.multiply(labels, features.dot(w)).shape)
        # print("temp1111",temp.shape)
        temp[temp>0] = 1
        # print("temp",temp)
        # print("temp.shape",temp.shape)



        temp_sd = -np.multiply(labels, features)
        # print("temp_sd.shape",temp_sd.shape)

        subgradient = np.multiply(temp, temp_sd)
        # subgradient = temp.T*temp_sd
        # print("subgradient",subgradient)
        # print("subgradient.shape",subgradient.shape)


        subgradient = np.sum(subgradient, axis = 0)
        subgradient = (1/minibatch_size)*subgradient

        # print("subgradient",subgradient)
        # print("subgradient.shape",subgradient.shape)
        return(value,subgradient.T)

    else:
        raise ValueError("The argument \"order\" should be 0 or 1")
