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


#Method : process_detection
#Purpose: Draw the bouding box on face detections
#Input  : 
#       frame, the image on which bounding box needs to be drawn
#       detections, actual detections as an array
#       confidence, probability of an actual face.
def process_detection(frame, detections, conf_threshold = 0.5):
    bboxes = []
    # Identify the size of the incoming image.
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > conf_threshold:
            x1 = int(detections[0,0,i,3]*frame_w)
            y1 = int(detections[0,0,i,4]*frame_h)
            x2 = int(detections[0,0,i,5]*frame_w)
            y2 = int(detections[0,0,i,6]*frame_h)
            bbboxes.append([x1,y1,x2,y2])
            bb_line_thickness = max(1,int(round(frame_h/200)))
            # Now draw the box
            cv2.rectangle((x1,y1),(x2,y2),(0,255,0),bb_line_thickness, cv2.LINE_8)
    return frame,bboxes



# Main starts here
img_file_buffer = st.file_uploader("Choose a file or camera", type=['jpg','jpeg','png'])
st.text('OR')
url = st.text_input('Enter URL')

# Give an option
option = st.selectbox('What would you like to do?',
                    ('Face Detection', 'Object Identification'))
st.write('You selected:', option)


if img_file_buffer is not None:
    # Read the image and convert into opencv
    if(option == 'Object Identification'):
        # Call the DNN model on the image
        image = np.array(Image.open(img_file_buffer))
        net, class_names = load_densenet_121()
        detections = classify(net,image, class_names)
        st.image(image)
        header(detections)
    else:
        # Now detections code
        raw_bytes = np.asarray(bytearray(img_file_buffer.read()),dtype=np.uint8)
        image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        placeholders = st.columns(2)

        # Show first image as is
        placeholders[0].image(image, channels = 'BGR')
        placeholders[0].text("Input image")
        net = load_res10_model()
        detections = detect_face(net,image)
        out_image, _ = process_detection(image, detections)

        # Now the image with BB
        placeholders[1].image(out_image,channels='BGR')
        placeholders[1].text("Output Image")

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

