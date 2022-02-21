'''
Created by Daniel-Iosif Trubacs for the UOS AI SOCIETY on 2 January 2022.
A simple module created to simplify the functions needed to evaluate the model. Used only for sign language
containing the english alphabet (except J and Z). To use this module just import the evaluate function
'''

import numpy as np
import pickle
from tensorflow import keras

#real labels
sign_labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
#print(len(sign_labels))


# functions used for prediction
# returning the max element representing the highest probability for a letter given by the model
def max_index(a):
   for i in range(len(a)):
     if a[i] == max(a):
       return i

'''
A functions to make predictions using a trained sr model. 
Inputs : model - a trained sr model
         image - a gray image of dimensions (28,28)
Outputs: A letter from the english alphabet (with the exception of J and Z) representing the element with the 
         highest probability assigned by the model

'''
def evaluate(model,image):
    image = np.reshape(image,(1,28,28,1))
    prediction = np.reshape(model.predict(image),(25,))
    real_index = max_index(prediction)
    #print(real_index)
    return sign_labels[real_index]



