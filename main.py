#Author           : Sateesh Kalidas
#Date              : 18/June/2023
#Purpose         : A fun project to detect images based on CNN
#Requirements :
# 1    The application must run on iOS
# 2    The application must be hosted on Azure cloud
# 3    The application should  be able to open camera to detect the image
# 4    The app must provide options for user to choose between image inference or face detection.

##### Imports #############################################
from requests.models import MissingSchema
import streamlit as st
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
import base64


##### Constants ##############################################
INPUT_WIDTH      = 640
INPUT_HEIGHT     = 640
CONFIDENCE_LIMIT = 0.45
SCORE_LIMIT      = 0.50
NMS_LIMIT        = 0.45
THICKNESS        = 4
BLUE             = (255,178,50) 
FONT_FACE        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE       = 3
BLACK            = (0,0,0)
YELLOW           = (0,255,255)

#### Helper functions. ##################################################
#Method  : Header
#Purpose : Print the inteferred image along with % confidence
#Input   : Data from the classify method
#Output  : None
def header(text):
     st.markdown(
             '<p style="background-color:#0066cc;color:#33ff33;font-size:24px;'
             f'border-radius:2%;", align="center">{text}</p>',
             unsafe_allow_html=True)

#Method : draw_label
#Input  :
#       input_image -> Image on which label needs to be drawn
#       label       -> Hmm, label
#       left        -> Top left
#       top
def draw_label(input_image,label,left,top):
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim,baseline = text_size[0], text_size[1]
    cv2.rectangle(input_image, 
                    (left,top),
                    (left + dim[0], top + dim[1] + baseline),
                    BLACK,
                    cv2.FILLED)
    cv2.putText(input_image,
                label,
                (left,top + dim[1]),
                FONT_FACE,
                FONT_SCALE,
                YELLOW,
                THICKNESS,
                cv2.LINE_AA)

##### DNN model loader methods ######################################################

#Method  : load_yolov5_model
#Purpose : Load the yolov5 model.
#Input   : None
#Output  : None
def load_yolov5():
    net = cv2.dnn.readNet('yolov5s.onnx')
    return net

#Method  : load_res10_model
#Purpose : Load Res Net model
#Input   : None
#Output  : the loaded model
def load_res10_model():
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net

#    """Method : load_densenet_121. """
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

##### Using the DNN models ############################################


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

#Method : process_face_detection
#Purpose: Draw the bouding box on face detections
#Input  : 
#       frame, the image on which bounding box needs to be drawn
#       detections, actual detections as an array
#       confidence, probability of an actual face.
#       blur, True or False
def process_face_detection(frame, detections, blur,conf_threshold = 0.5):
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
            bboxes.append([x1,y1,x2,y2])
            bb_line_thickness = max(1,int(round(frame_h/200)))
            # Now draw the box
            cv2.rectangle(frame,(x1,y1),(x2,y2), (0,255,0),8)
            label = 'Confidence: %.4f' % confidence
            font = cv2.FONT_HERSHEY_SIMPLEX
            label_size,base_line = cv2.getTextSize(label,font,fontScale = 1.5,thickness = 4)
            cv2.rectangle(frame,
                    (x1,y1 - label_size[1]),
                    (x1+label_size[0],y1 + base_line),
                    (255,255,255),
                    cv2.FILLED)
            cv2.putText(frame,label,(x1,y1),font,1.5, (0,0,0))
            if(blur == True):
                (h,w) = frame.shape[:2]
                box = detections[0,0,i,3:7]*np.array([w,h,w,h])
                (x1,y1,x2,y2) = box.astype('int')
                frame_temp = frame[y1:y2,x1:x2]
                frame_temp = blur_face(frame_temp,1)
                frame[y1:y2,x1:x2] = frame_temp
    return frame,bboxes

#Method : For an imput image, simple blur the image and return a blurred copy
#Input  : face_only  -> the input to blur
#         factor -> how much blurring is needed
#output : bluerred -> actual blurred image
def blur_face(face_only,factor=3):
    h, w = face_only.shape[:2]

    if(factor < 1):
        factor =1

    if(factor > 5):
        factor = 5

    # New blurring kernel
    w_k = int(w/factor)
    h_k = int(h/factor)

    # Make kernel an odd one
    if(w_k%2 == 0):
        w_k += 1
    if(h_k%2 ==0):
        h_k += 1

    blurred = cv2.GaussianBlur(face_only,(int(w_k),int(h_k)),0,0)
    return blurred

#Method : preprocess_object_detection
#Purpose:
#Input  : 
#         image -> input image
#         net   -> DNN model
#Output
#        output -> detection.
def preprocess_object_detection(input_image,net):
    blob = cv2.dnn.blobFromImage(input_image,
            1/255,
            (INPUT_WIDTH, INPUT_HEIGHT),
            [0,0,0],
            1,
            crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)
    return outputs


#Method: postprocess_object_detection(input_image, outputs)
#Input :
#        input_image -> Input on which annotations to be drawn
#        outputs     -> Annotated with BB
#Output:
#        input_image -> With annotations and BB for object detections
def postprocess_object_detection(input_image, outputs):
    # Lists to hold values while unwrapping
    class_ids   = []
    confidences = []
    boxes       = []

    # Rows of detections
    rows = outputs[0].shape[1]
    image_height, image_width = input_image.shape[:2]

    # Resize
    x_factor = image_width/INPUT_WIDTH
    y_factor = image_height/INPUT_HEIGHT

    # Go through all detections.
    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]

        if(confidence >= CONFIDENCE_LIMIT):
            classes_scores = row[5:]
            class_id = np.argmax(classes_scores)

            if(classes_scores[class_id] > SCORE_LIMIT):
                confidences.append(confidence)
                class_ids.append(class_id)
                cx,cy,w,h = row[0],row[1], row[2],row[3]
                left   = int((cx-w/2)*x_factor)
                top    = int((cy-h/2)*y_factor)
                width  = int(w * x_factor)
                height = int(h*y_factor)
                box = np.array([left,top,width,height])
                boxes.append(box)
    indices = cv2.dnn.NMSBoxes(boxes,confidences,CONFIDENCE_LIMIT, NMS_LIMIT)
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        cv2.rectangle(input_image,
                (left,top),
                (left+width,top+height),
                BLUE,
                3*THICKNESS)
        label = "{}:{:.2f}".format(classes[class_ids[i]],confidences[i])
        draw_label(input_image, label, left,top)
    return input_image




# Main starts here
st.title("Image classifier using DenseNet 121, Face recognition using ResNet and Text recognition using EAST")
img_file_buffer = st.file_uploader("Choose a file or camera", type=['jpg','jpeg','png'])
st.text('OR')
url = st.text_input('Enter URL only for Image classification.')

# Give an option
option = st.selectbox('What would you like to do?',
                    ('Object Identification','Face Detection', 'Image Inference','Text Detection'))

# Command line processing.
if img_file_buffer is not None:
    if(option == 'Object Identification'):
        image = np.array(Image.open(img_file_buffer))
        classesFile = 'coco.names'
        classes = None
        with open(classesFile,'rt') as f:
            classes = f.read().rstrip('\n').split('\n')
        net = load_yolov5()
        obj_detections = preprocess_object_detection(image,net)
        out_image = postprocess_object_detection(image.copy(),obj_detections)
        resized_image = cv2.resize(out_image,None,fx=0.4,fy=0.4)
        st.image(resized_image)
    if(option == 'Image Inference'):
        image = np.array(Image.open(img_file_buffer))
        # Call the DNN model on the image
        net, class_names = load_densenet_121()
        detections = classify(net,image, class_names)
        st.image(image)
        header(detections)
    if(option == 'Face Detection'):
        # Now detections code
        raw_bytes = np.asarray(bytearray(img_file_buffer.read()),dtype=np.uint8)
        image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        placeholders = st.columns(3)

        # Show first image as is
        placeholders[0].image(image, channels = 'BGR')
        placeholders[0].text("Input image")
        net = load_res10_model()
        face_detections = detect_face(net,image)
        out_image, _ = process_face_detection(image, face_detections,False)

        # Now the image with BB
        placeholders[1].image(out_image,channels='BGR')
        placeholders[1].text("Output Image")
        option2 = st.selectbox('What would you like blur the image for privacy?',
                    ('Yes','No'))
        #Start the blur code.
        if(option2 == 'Yes'):
            out_image, _ = process_face_detection(image, face_detections,True)
            # Identify the area of the face and print it out
            placeholders[2].image(out_image, channels='BGR')
            placeholders[2].text("Blurred image")
        else:
            out_image, _ = process_face_detection(image, detections,False)
            # Identify the area of the face and print it out
            placeholders[2].image(out_image, channels='BGR')
            placeholders[2].text("Un-Blurred image")
    if(option == 'Text Detection'):
        image = np.array(Image.open(img_file_buffer))
        image = np.array(Image.open(img_file_buffer))
        net = cv2.dnn_TextDetectionModel_EAST('frozen_east_text_detection.pb')
        conf_thresh = 0.8
        nms_thresh  = 0.4
        inputSize = (320,320)
        net.setConfidenceThreshold(conf_thresh).setNMSThreshold(nms_thresh)
        net.setInputParams(1.0,inputSize,(123.68,116.78,103.94), True)
        placeholders = st.columns(2)
        # Show first image as is
        placeholders[0].image(image, channels = 'RGB')
        placeholders[0].text("Input image")
        imEAST = image.copy()
        boxesEAST , confsEAST = net.detect(image)
        cv2.polylines(imEAST,boxesEAST, isClosed = True, color=(255,0,255), thickness=4)
        # Now the image with BB
        placeholders[1].image(imEAST,channels='RGB')
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

