import cv2

import asyncio
import io
import glob
import os
import sys
import time
import uuid
import requests
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, SnapshotObjectType, OperationStatusType
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

DATABASE_URL=os.environ.get('DATABASE_URL')

engine = create_engine(DATABASE_URL)
db = scoped_session(sessionmaker(bind=engine))

def addtransac(email,predicted):
  db.execute(""" INSERT INTO "Transcript" (user_email,prediction,timestamp)
                    VALUES (:email,:pre,:time);
                    COMMIT;""",{"email":email,"pre":predicted,"time":int(time.time())})
  print("Added transac")

def getStudentID(PersonID):
    res=db.execute(f"""
                    SELECT "School_ID" FROM "user" WHERE "PersonID" = '{PersonID}';
                    """).fetchall()
    return res[0][0]

def predict(faceimg):
    filename=uuid.uuid4().hex
    cv2.imwrite("facesave/"+filename+".jpg", faceimg)
    test_image_array = glob.glob("facesave/"+filename+".jpg")
    faceimg = open(test_image_array[0], 'r+b')
    face_ids = []
    faces = face_client.face.detect_with_stream(faceimg)
    if (len(faces)==0):
        return "NoFaceFound",filename
    elif(len(faces)>1):
        return "TooManyPeople",filename
    else:
        results = face_client.face.identify([faces[0].face_id], PERSON_GROUP_ID)
        if not results:
            return "ERROR Can't Find Faces",filename
        person=results[0]
        if(len(person.candidates)==0):
            return "ERROR Can't Match Face",filename
        userid = getStudentID(person.candidates[0].person_id)
        confident=person.candidates[0].confidence
        if(confident<=0.61):
            return "ERROR Can't Match Face",filename
        addtransac("DesktopApp",userid)
        return userid,filename

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default_official.xml')
print("Loaded cascade")
# To capture video from webcam. 
cap = cv2.VideoCapture(0)
print("Started cv2.VideoCapture")
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')
KEY = os.environ['FACE_SUBSCRIPTION_KEY']
ENDPOINT = os.environ['FACE_ENDPOINT']
PERSON_GROUP_ID=os.environ['PERSON_GROUP_ID']
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

#Set Full Screen
print("Set Window")
cv2.startWindowThread()
# cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

steakcount=0
while True:
    # Read the frame
    _, img = cap.read()
    print("cab read")
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("gray")
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    print("face_cascade")
    # Draw the rectangle around each face
    countfaces=0
    for (x, y, w, h) in faces:
        if(w*h>=(280*280)):
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            countfaces=countfaces+1
    print("countfaces : ",countfaces)
    if (countfaces>1):
        steakcount=0
        img=cv2.putText(img, "There more than one persorn!",(50, 50),cv2.FONT_HERSHEY_SIMPLEX,
                    1,(255, 0, 0),2,cv2.LINE_AA)
        continue
    elif(countfaces==1):
        steakcount=steakcount+1
        if (steakcount<12):
            img=cv2.putText(img, "STREAK="+str(steakcount),(50, 50),cv2.FONT_HERSHEY_SIMPLEX,
                    1,(255, 0, 0),2,cv2.LINE_AA)
            cv2.imshow('window', img)
            continue
        else:
            steakcount=0
        respond,filename=predict(img[y:y+h,x:x+w])
        print(respond,filename)
        img=cv2.putText(img, respond,(50, 50),cv2.FONT_HERSHEY_SIMPLEX,
                    1,(255, 0, 0),2,cv2.LINE_AA)
    else:
        steakcount=0

    # Display
    cv2.imshow('window', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()