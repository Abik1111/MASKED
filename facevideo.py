import argparse
import numpy as np
import imutils
from imutils.video import FileVideoStream
from imutils.video import VideoStream
import time
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2

#specifying the minimum confidence required for the detected face to be valid
min_confidence=0.9

#loading the model for detection of face for use in open cv
net=cv2.dnn.readNetFromCaffe('deploy.prototxt.txt','res10_300x300_ssd_iter_140000.caffemodel')

#loading the model that we trained to seperate between the masked and unmasked face
model=keras.models.load_model('trained_mask.h5')

###stating the video stream camera and allow the camera stream 
##to warm up
print('[INFO] starting video stream')
vs=VideoStream(src=0).start()
time.sleep(2.0)

#creating the tracker to allow following of the objects even when prople try to hide
trackers=cv2.MultiTracker_create()

# function that predicts using the loaded model
def prediction(ROI):
    # if len(ROI)==0:#check if nothing is passed argument
    #     return False
    #convert the BGR image to RGB
    temp=cv2.cvtColor(ROI,cv2.COLOR_BGR2RGB)
    #resize to the size that is appropriate for MobileNetV2
    temp=cv2.resize(temp,(224,224))
    #preprocess the image
    temp=preprocess_input(temp)
    #increaing the dimension to make batchsize=1 so that model can take the image as input
    temp=np.expand_dims(temp,axis=0)
    pred_val=model.predict(temp)
    #return the predicted value
    return pred_val

#this function tries to determine whether the tracker is already tracking the object
def already_existing(single,multiple):
    if len(single)==0:
        return False
    #get the value of the bounding rectangle
    (startX,startY,endX,endY)=single.astype(int)
    # calculate its area
    base_area=(endX-startX)*(endY-startY)
    # compare it with all the tracked objects
    for one in multiple:
        #get the origin, width and height of the tracked box
        (x,y,w,h)=[int(dim) for dim in box1]
        #get the corners to calculate the common area
        bnd_startX=startX if startX>=x else x
        bnd_startY=startY if startY>=y else y
        bnd_endX=endX if endX<=(x+w) else (x+w)
        bnd_endY=endY if endY<=(y+h) else (y+h)
        # calculate the common area
        area=(bnd_endX-bnd_startX)*(bnd_endY-bnd_startY)
        # if the overlapping area is greater than 50% of the box then it is considered as the same object
        if area>(0.5 * base_area):
            return True



#forming the blob and getting the images to loop over each
while True:
    frame=vs.read()
    # resizing the frame i.e is the image from video
    frame=imutils.resize(frame,width=500)
    #get the shape of the frame
    (h,w)=frame.shape[:2]

    # creating blob from the image
    blob=cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1, (300,300),(104.0,177.0,123.0))

    #setting the blob as the imput to the network that detects faces
    net.setInput(blob)

    #the predicted values
    detections=net.forward()

    #get the new locations of boxes from the tracker object
    (success,boxes)=trackers.update(frame)
    print(success)
    #   drawing the boxes according to their new location
    for box1 in boxes:
        (x,y,w,h)=[int(dim) for dim in box1]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    # this loop goes through each of the detection from the face detector
    for i in range(0, detections.shape[2]):

        #the probability that the detected part of the image is a face
        confidence=detections[0,0,i,2]

        # comparing the probability with the threshold for further work
        if confidence > min_confidence:
            print("Object",i)
            #get the coordinates of the bounding box
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype(int)
            
            # predict whether the Region of Interest consist of people wearing mask
            pred_val=prediction(frame[startY:endY,startX:endX])
            pred_val=np.ravel(pred_val).item()

            # Threshold value for determining whether people are wearing mask
            if pred_val<0.6:
                # skip the detection if the object is already being tracked
                if already_existing(box,boxes):
                    print("Skipped")
                    continue
                # if not already being tracked, track it
                tracker=cv2.TrackerCSRT_create()
                #add the tracker to the list of tracker
                trackers.add(tracker,frame,(startX,startY,endX-startX,endY-startY))
    
    # output the frame with the rectangles
    cv2.imshow("Output",frame)
    #check if any key is pressed
    key=cv2.waitKey(1) 
    
    # if 'q ' is pressed then exit
    if key==ord('q'):
        break
    
cv2.destroyAllWindows()
vs.stream.release()

