'''
Created by Daniel-Iosif Trubacs for the UoS AI society on 2 January 2022. To be used with the SRModel and
the HandDectection modules. Using a trained Convolutional NN (trained on the SIGN_MNIST DATA) and a
Hand Detection algorithm to recognize sign languages.
'''


# importing the necessary libraries
import mediapipe as mp
import cv2 as cv
from tensorflow import keras
from ModelEvaluation import evaluate
from HandDetection import HandDetection
import pickle
from matplotlib import pyplot as plt

# the font used for showing different coordinates and positions on the image
font = cv.FONT_HERSHEY_SIMPLEX

#loading the trained model
sr_model = keras.models.load_model('sign_recongiton_model')

#checking if the model has been loaded succesfully
if sr_model:
    print("The model has been loaded  succesfully ")

# getting the live feed from a web site (created using IP web cam on adroid)
capture = cv.VideoCapture('http://10.14.132.177:8080/video')
#checking if the web site is created
if capture:
    print("Live feed set")

#testing an image
#with open('test_image_0', 'rb') as img:
   #img_model = pickle.load(img)


start = True
# starting the detection
while start:
    # getting the original frame and storring the objects found
    A, frame = capture.read()


    # changing to RGB for hand detection algorithm (required by media pipe)
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    '''
    The Hand hand detection algorithm. It takes as input an RGB image and returns a gray image of size 28x28.
    This size is required to feed it into the neural net
    '''
    try:
     Hand = HandDetection(img)
     img_model = Hand.highlight(show_hand=True, show_landmarks=True)
     # feed the detected hand into the model and make prediction
     prediction = evaluate(sr_model, img_model)
     #print(prediction)
    except:
     print("NO HAND DETECTED OR ALGORTIHM FAILED")
     prediction = 'NO HAND DETECTED'


    #resizing the image (easier to show on the screen)
    frame = cv.resize(frame,(1500,750))

    #showing the prediction on the image
    cv.putText(frame,'PREDICTED SIGN: '+prediction,(50,50),font,1,(0,0,255),2,cv.LINE_AA)

    cv.imshow("Live feed", frame)
    if cv.waitKey(1) == ord('q'):
       break


capture.release()
cv.destroyAllWindows()