'''
Sign Language Translation.
This model has been created by Daniel-Iosif Trubacs for the UOS AI SOCIETY on 2 January 2022
Aiming to use Conv-Nets for sign language recognition. The trained model is used together with a hand
detection algorithm on a live feed.
For easier reading, the code is split in parts. Each part is commented at the beginning and is separated by
the other parts by 3 empty lines.
'''


#importing the necessary libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
import pickle

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
train_labels = train_data[:, 0]
test_labels = test_data[:, 0]

# dividing by 255 to have all elements in the tensor between 0 and 1
train_images = train_data[:, 1:]/255
test_images = test_data[:, 1:]/255

# reshaping the train images to 28x28 matrices (representing a image in gray scale)
train_images = np.reshape(train_images, (27455, 28, 28))
test_images = np.reshape(test_images, (7172, 28, 28))



'''
Tokenizing the data (chancing from scalars to vectors with binary elements) for faster and better training.
'''
# tokenizing the labels
def one_hot(a):
  one_hot = np.zeros((len(a), 25))
  for i in range(len(a)):
    one_hot[i][a[i]] = 1
  return one_hot

test_labels = one_hot(test_labels)
train_labels = one_hot(train_labels)

# reshaping the train and test data (28 x 28 images) to have 1 color channel
train_images = np.reshape(train_images, (27455, 28, 28, 1))
test_images = np.reshape(test_images, (7172, 28, 28, 1))

# taking some validation data
val_images = test_images[6000:]
val_labels = test_labels[6000:]

# correcting the test data
test_images = test_images[:6000]
test_labels = test_labels[:6000]



'''
Creating a Convolutional model to solve the sign language recognition problem. This model used the Keras API.
'''
# building the basics of the model
def build_model(Input_Shape):
    # input
    input_layer = Input(shape=Input_Shape)

    # layers: Conv2D -> Dropout -> BatchNorm -> LeakyRelu
    x = Conv2D(filters=12, kernel_size=9, strides=1, padding='same',
               input_shape=(28, 28, 1))(input_layer)
    x = Dropout(rate=0.2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)

    # layers: Conv2D -> Dropout -> BatchNorm -> LeakyRelu
    x = Conv2D(filters=24, kernel_size=5, strides=1, padding='same')(x)
    x = Dropout(rate=0.2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # layers: Conv2D -> Dropout -> BatchNorm -> LeakyRelu
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = Dropout(rate=0.2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # layers: Conv2D -> Dropout -> BatchNorm -> LeakyRelu
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = Dropout(rate=0.4)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # flattening the data to feed into a FC net
    x = Flatten()(x)

    # FC layers
    # normal dense layers using relu activation
    x = Dense(units=64, activation='relu', kernel_regularizer=l2(0.3))(x)
    x = BatchNormalization()(x)
    x = Dense(units=32, activation='relu', kernel_regularizer=l2(0.3))(x)

    # using softmax for returning a probability distribution
    x = Dense(units=25, activation='softmax')(x)

    # the output layer
    output_layer = x

    # compiling and returning the model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
    return model

# Building the model
sr_model = build_model(Input_Shape=(28,28,1))
sr_model.summary()



'''
Training the model using the data from before and showing a history object of the training epochs.
'''
# training the model
n_epochs = 10
history = sr_model.fit(train_images, train_labels, epochs=n_epochs,  batch_size=216, validation_data=(val_images, val_labels))

# showing a plot of the model
plt.figure(num=1, figsize=[12, 8])
plt.title("Sign Language Recognition model")
plt.subplot(211)
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1, n_epochs+1)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.ylabel('Loss')
plt.legend()
plt.subplot(212)
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1,n_epochs+1)
plt.plot(epochs, accuracy, 'g', label='training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# checking the accuracy of the model on the test data
test_loss, test_acc = sr_model.evaluate(test_images, test_labels)
print("The test loss is:", test_loss)
print("The test accuracy is:", test_acc)




'''
This part is optional. If you want to save the model set saving to True and if you want a short sketch of the model set diagram to true
'''
# saving the model
saving = True
if saving:
 sr_model.save("sign_recongiton_model")
# a quick sckecth
diagram = True
if diagram:
 plot_model(sr_model, to_file='model.png', show_shapes=True, show_layer_names=True)