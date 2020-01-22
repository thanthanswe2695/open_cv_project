###Before train an image , need to be downloaded images
import os
from PIL import Image
import numpy as np  
import pickle
import cv2

# /home/curiousgirl/Desktop/face_recognition_identification/open_cv/src/images
# 
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
# BASE_DIR=os.path.dirname("/home/curiousgirl/Desktop/face_recognition_identification/open_cv")
print(BASE_DIR)


image_dir=os.path.join(BASE_DIR,'src/cascades/images')
print(image_dir)

face_cascade=cv2.CascadeClassifier('/home/curiousgirl/Desktop/face_recognition_identification/open_cv/src/cascades/data/haarcascade_frontalface_alt2.xml')

recognizer=cv2.face.LBPHFaceRecognizer_create()

current_id=0
label_ids={}

x_train=[]
y_labels=[]

for root,dirs,files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
			path=os.path.join(root,file)
			# label from directory
			label=os.path.basename(root).replace(" ","-").lower()
			# print(label,path)
			
			if not label in label_ids:
				label_ids[label]=current_id
				current_id+=1
			id_=label_ids[label]
			# print(label_ids)

			# y_labels.append(label)  #some number 
			# x_train.append(path)  # to verify this image , turn into numpy array,gray
	##Training images into numpy array
			print("Path : ",path, type(path))
			pil_img=Image.open(path).convert("L") 
			size=(550,550)
			final_image=pil_img.resize(size,Image.ANTIALIAS)
			image_array=np.array(final_image)
			print(image_array)
	##Region of interest in training data
			faces=face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5)
			for (x,y,w,h) in faces:
				roi=image_array[y:y+h,x:x+w]
				x_train.append(roi)
				y_labels.append(id_)

# print('Y labels  :' ,y_labels)
# print('X train   : ' ,x_train)

##Saving labels using pickle
with open("labels.pkl","wb") as f:
	pickle.dump(label_ids,f)

#Train the facial recognizer

recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainer.yml")