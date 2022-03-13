'''
Sign Language Translation.
Created by Daniel-Iosif Trubacs for the UOS AI SOCIETY on 2 March 2022
This model represent a better version of the SignRecModel which accounts for both letters and digits.
The data data has been processed in data_processing module and saved to numpy arrays with pickle
The model uses residual conv nets.
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



# loading the necesary data.
with open('data_images','rb') as handle:
    images = pickle.load(handle)
with open('data_labels','rb') as handle:
    labels = pickle.load(handle)
print("The data has been loaded")
print("The shape of the images tensor is:",images.shape)
print("The shape of the labels tensor is:",labels.shape)



#spliting the data into train data and test data
train_images = images[:42000]
train_labels = labels[:42000]
test_images = images[42000:]
test_labels = labels[42000:]
print("The shape of the training data is:")
print("images:",train_images.shape)
print('labels',train_labels.shape)
print("The shape of the test data is:")
print('images:',test_images.shape)
print('labels',test_labels.shape)




'''
Creating a Convolutional model to solve the sign language recognition problem. This model used the Keras API.
'''
# building the basics of the model
def build_model(Input_Shape):
    # input
    input_layer = Input(shape=Input_Shape)

    # layers: Conv2D -> Dropout -> BatchNorm -> LeakyRelu
    x = Conv2D(filters=64, kernel_size=9, strides=1, padding='same',
               input_shape=(28, 28, 1))(input_layer)
    x = Dropout(rate=0.2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)

    # layers: Conv2D -> Dropout -> BatchNorm -> LeakyRelu
    x = Conv2D(filters=128, kernel_size=5, strides=1, padding='same')(x)
    x = Dropout(rate=0.2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # layers: Conv2D -> Dropout -> BatchNorm -> LeakyRelu
    x = Conv2D(filters=214, kernel_size=3, strides=1, padding='same')(x)
    x = Dropout(rate=0.2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # layers: Conv2D -> Dropout -> BatchNorm -> LeakyRelu
    x = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(x)
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
    x = Dense(units=36, activation='softmax')(x)

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
history = sr_model.fit(train_images, train_labels, epochs=n_epochs,  batch_size=216, validation_split=0.1)

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
 sr_model.save("sign_recongiton_model_version")
# a quick sckecth
diagram = True
if diagram:
 plot_model(sr_model, to_file='model.png', show_shapes=True, show_layer_names=True)







