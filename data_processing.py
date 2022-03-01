'''
Data processing for sign language recognition. The data used has been taken from: https://www.kaggle.com/datamunge/sign-language-mnist.
(all letters from the english alphabet except J and Z) and https://github.com/ardamavi/Sign-Language-Digits-Dataset (for the digits 0-9)
All the data has been transformed to numpy arrays of dimension 28x28 (representing a image in gray scale) and the labels are arrays
of size (36,) (corresponding to the sign_labels shown  below). The data has been saved with pickle and is ready to use for training a model.
'''

#importing the necessary libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
from sklearn.utils import shuffle

##real labels
sign_labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V',
               'W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']

# functions used for prediction
# returning the max element representing the highest probability for a letter given by the model
def max_index(a):
   for i in range(len(a)):
     if a[i] == max(a):
       return i

'''
Reading the data from a CSV file and transforming into a numpy array to feed into the neural net
'''
# reading the raw data
test_data = pd.read_csv('sign_mnist_test.csv')
train_data = pd.read_csv('sign_mnist_train.csv')

# transforming to numpy array
train_data = train_data.to_numpy()
test_data = test_data.to_numpy()

# separating the label from the images
train_labels = train_data[:,0]
test_labels = test_data[:,0]

# dividing by 255 to have all elements in the tensor between 0 and 1
train_images = train_data[:,1:]/255
test_images = test_data[:,1:]/255




# reshaping the train images to 28x28 matrices (representing a image in gray scale)
train_images = np.reshape(train_images,(27455,28,28))
test_images = np.reshape(test_images,(7172,28,28))

# adding all the images representing letters
data_images = np.concatenate((train_images,test_images),axis=0)
data_labels = np.concatenate((train_labels,test_labels),axis=0)

# tokenizing the labels
def one_hot(a):
  one_hot = np.zeros((len(a), 36))
  for i in range(len(a)):
    one_hot[i][a[i]] = 1
  return one_hot
data_labels = one_hot(data_labels)


# creating the dataset for digits
# reading the first dataset corresponding to 0 to concatenate later
with open ('sign_images_0', 'rb') as handle:
     digits_data = pickle.load(handle)
     digits_data_labels = np.zeros((1500,36))
     for i in range(1500):
       digits_data_labels[i][26] = 1

# reading the rest of the data
for i in range(9):
  # reading the saved data
  with open('sign_images_'+str(i+1), 'rb') as handle:
    aux_data = pickle.load(handle)

  #creating the data labels:
  aux_data_labels = np.zeros((1500,36))
  for j in range(1500):
     aux_data_labels[j][26+i+1] = 1

  #concatening all the data from the digits found
  digits_data = np.concatenate((digits_data,aux_data),axis=0)
  digits_data_labels = np.concatenate((digits_data_labels,aux_data_labels),axis=0)


#concatening all the data we have (including letters and digits)
data_images = np.concatenate((data_images,digits_data),axis=0)
data_labels = np.concatenate((data_labels,digits_data_labels),axis=0)

#randomizing the data
data_images, data_labels = shuffle(data_images,data_labels)

#saving the data
with open('data_images','wb') as handle:
    pickle.dump(data_images,handle)
with open('data_labels','wb') as handle:
    pickle.dump(data_labels,handle)
print("The data has been saved")
print("The shape of the data is:")
print(data_images.shape)
print(data_labels.shape)

# checking to see if the data has been saved correctly. set Trial to True if you want to use it
trial = False
if trial:
 for i in range(10):
    #j = np.random.randint(0,49627)
    j = i
    plt.imshow(data_images[j],cmap='gray')
    plt.show()
    plt.clf()
    print(i," ",j," ",sign_labels[max_index(data_labels[j])])

plt.clf()

























