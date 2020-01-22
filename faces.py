import numpy as np 
import cv2
import pickle

face_cascade=cv2.CascadeClassifier('/home/curiousgirl/Desktop/face_recognition_identification/open_cv/src/cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('/home/curiousgirl/Desktop/face_recognition_identification/open_cv/src/cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('/home/curiousgirl/Desktop/face_recognition_identification/open_cv/src/cascades/data/haarcascade_smile.xml')

recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
print(recognizer)
labels={"person_name" :1}

with open("labels.pkl","rb") as f:
    open_labels=pickle.load(f)
    labels={v:k for k,v in open_labels.items()}

cap=cv2.VideoCapture(0)
while (True):
    #Capture frame by frame
    ret, frame=cap.read()
    #Using the classifier
    ## before we do the train , we convert an image into a gray
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ##scaleFactorr =1.5 might be more a little bit more accurate
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in faces:
        print(x,y,w,h)
    ### region of interest which means that my face is actually showing in that frame 
    #regional interest of gray
        roi_gray=gray[y:y+h,x:x+w]  ##[cord1-height,cord2-height]   
        #regional interest of color
        roi_color=frame[y:y+h,x:x+w]  
        ### Recognizer  used in deep learning model predict (keras ,tensorflow,pytorch)
        id_one, conf=recognizer.predict(roi_gray)
        print("id_  ",id_one)
        if conf >=45:    
            print(id_one)
            print(labels[id_one])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_one]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        img_item='my_image.png'
        cv2.imwrite(img_item,roi_color)


        ##Draw a rectangle
        color=(255,0,0) ##BGR 0-255
        stroke=2        #how thick draw a line
        end_cordinate_x=x+w         ##width
        end_cordinate_y=y+h         ##height
        cv2.rectangle(frame,(x,y),(end_cordinate_x,end_cordinate_y),color,stroke)

    #Display the resulting frame
    cv2.imshow('frame',frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

#When everything done , release the capture

cap.release()
cv2.destroyAllWindows()