from scipy.spatial import distance
#from pkg_resources import to_filename
from imutils import face_utils
#import playsound
import imutils
import dlib
import cv2
from threading import Thread
import numpy as np
import matplotlib.pyplot as plt
import sys
import urllib
import urllib.request
import time

import os
from twilio.rest import Client
#import argparse

import pygame

# Initialize pygame
pygame.mixer.init()

def sound_alarm(path):
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()

def stop_alarm():
    pygame.mixer.music.stop()

#def sound_alarm(path):
	#playsound.playsound(path)

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear


account_sid = ''	#TWILIO ACCOUNT SID
auth_token = ''	#TWILIO_AUTH_TOKEN


client = Client(account_sid, auth_token)


thresh = 0.25
frame_check = 20
ALARM_ON = False
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat") #Dat file is the crux of the code


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap=cv2.VideoCapture(0)
flag=0
list=[]
drowsy_count=0
total_drowsy=0
#blink=0
drowsiness_detected = False
cooldown_duration = 10 
last_detection_time = 0

while True:
	ret, frame=cap.read()
	frame = imutils.resize(frame, width=600)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)
	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)#converting to NumPy Array
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		
		if ear < thresh and not drowsiness_detected and time.time() - last_detection_time >= cooldown_duration:
			flag += 1
			#print (flag)
			list.append(int(flag))

			if flag >= frame_check:
				if not ALARM_ON:
					args = {"alarm": r"C:\Users\Aakash Praveen\Downloads\archivo_hPJRRnzJ.wav"}
					ALARM_ON = True
					if args["alarm"] != "":
						sound_alarm(args["alarm"])
						#t=Thread(target=sound_alarm,args=(args["alarm"],))
						#t.daemon = True
						#t.start()
				cv2.putText(frame, "***********************ALERT!*********************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "***********************ALERT!*********************", (10,425),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				print ("Drowsiness detected")
				drowsy_count = drowsy_count + 1
				total_drowsy = total_drowsy + 1
				#drowsy_check = drowsy_check + 1
				drowsiness_detected = True
				last_detection_time = time.time()
				if (total_drowsy % 3) == 0:
					message = client.messages.create(
  					from_='',#TWILIO NUMBER PROVIDED
  					body='This number has been provided as the emergency contact for the current user. Call and ensure their safety as they have been drowsy driving which could be fatal for their life.',
  					to=''#EMERGENCY CONTACT OF USER
					)
					print(message.sid)
					#drowsy_check = 0
					#ALARM_ON = True
		else:
			if drowsiness_detected:
				time.sleep(5)
				drowsiness_detected = False
				stop_alarm()
			flag = 0
			drowsy_count=0
			ALARM_ON = False
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 100),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame,f'Drowsy Count: {total_drowsy}',(300,150),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2,cv2.LINE_AA)
	cv2.imshow("GuardianEye : Drowsiness Detection and Alert System", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

cv2.destroyAllWindows()
cap.release() 

y=np.array(list)
x=np.arange(len(list))

print('Total Drowsy Count : ',total_drowsy)

plt.title("Drowsiness Values")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.plot(x,y,color="red")
plt.show()
