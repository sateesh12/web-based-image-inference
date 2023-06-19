#Author           : Sateesh Kalidas
#Date              : 18/June/2023
#Purpose         : A fun project to detect images based on CNN
#Requirements :
# 1    The application must run on iOS
# 2    The application must be hosted on Azure cloud
# 3    The application should  be able to open camera to detect the image
# 4    The app must provide options for user to choose between image inference or face detection.

from requests.models import MissingSchema
import streamlit as st
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
import base64

#OK print('All imports completed')

#Give the application a name 
st.title("Image classifier")


#Method  : load_res10_model
#Purpose : Load Res Net model
#Input   : None
#Output  : the loaded model
def load_res10_model():
    modelFile = "res10_300x200_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffee(configFile, modelFile)
    return net



#    """Method : load_model. """
#    """Purpose: Loads the DNN model DenseNet 121 which has been trained by Caffe."""
#    """Input  : None."""
#    """Return : None. """
def load_densenet_121():

    # Read the ImageNet class names.
    with open('classification_classes_ILSVRC2012.txt','r') as f:
        image_net_names = f.read().split('\n')
        #OK print('image net names are:',image_net_names , '\n')
    
    #As there are 1000 objects that can be detected, simply it a bit to pick the first one in the row
    class_names = [name.split(',')[0] for name in image_net_names]
    #OK print('class names are', class_names)

    #Load the dense net trained model.
    model = cv2.dnn.readNet(
            model = 'DenseNet_121.caffemodel',
            config = 'DenseNet_121.prototxt',
            framework='Caffe')
    return model, class_names


#    """Method      : Classifiy the images """
#    """Purpose     : Classify the images """
#    """Inputs      : """
#    """model       : The DenseNet Model"""
#   """image       : Image to classify """
#   """class_names : High level class names """ 
#   """ Output     : """
#   """ out_name   : Name of the object identification with highest percentage. """
def classify(model, image, class_names):
    # Remove alpha channel from the image
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)


    # Make a binary blob of the input image.
    blob = cv2.dnn.blobFromImage(
            image = image,
            scalefactor = 0.017,
            size=(224,224),
            mean = (104,117,123))


    # Send blob into NN
    model.setInput(blob)
    outputs = model.forward()

    final_outputs = outputs[0]
    final_outputs = final_outputs.reshape(1000,1)
    
    # Get the identified label
    label_id = np.argmax(final_outputs)

    # Do a softmax
    probs = np.exp(final_outputs) / np.sum(np.exp(final_outputs))
    final_prob = np.max(probs)*100

    # Map the max value to a human recognizable name
    out_name = class_names[label_id]
    out_text = f"Class: {out_name}, Confidence: {final_prob:.1f}%"
    return out_text

#Method  : Header
#Purpose : Print the inteferred image along with % confidence
#Input   : Data from the classify method
#Output  : None
def header(text):
     st.markdown(
             '<p style="background-color:#0066cc;color:#33ff33;font-size:24px;'
             f'border-radius:2%;", align="center">{text}</p>',
             unsafe_allow_html=True)


#Method  : detect_face
#Purpose : Annotate a face with a bounding box
#Input   : DNN model,image 
#Output  : detections
def detect_face(net,image):
    blob = cv2.dnn.blobFromImage(image,
            1.0,
            (300,300),
            [104,117,123],
            False,
            False)
    net.setInput(blob)
    detections = net.forward()
    return detections


# Main starts here
net, class_names = load_densenet_121()
img_file_buffer = st.file_uploader("Choose a file or camera", type=['jpg','jpeg','png'])
st.text('OR')
url = st.text_input('Enter URL')

# Give an option
option = st.selectbox('What would you like to do?',
                    ('Face Detection, Object Identification'))
st.write('You selected': option)


if img_file_buffer is not None:
    # Read the image and convert into opencv
    image = np.array(Image.open(img_file_buffer))
    st.image(image)

    # Call the DNN model on the image
    detections = classify(net,image, class_names)
    header(detections)

elif url != '':
    try:
        response = requests.get(url)
        image = np.array(Image.open(BytesIO(response.content)))
        st.image(image)
        # Call the DNN model on the image
        detections = classify(net,image, class_names)
        header(detections)
    except MissingScheme as err:
        st.header("Invalid URL, Try Again!")
        print(err)
    except UnidentifiedImageError as err:
        st.header("URL has no image, Try Again!")
        print(err)

