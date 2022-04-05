import cv2
import os
from datetime import datetime

now = datetime.now() # current date and time

date_time = now.strftime("%Y%m%d_%H%M%S")
print("time:",date_time)

facesFolder = 'faces/'+date_time

if not os.path.exists(facesFolder):
    print('Carpeta creada: '+facesFolder)
    os.makedirs(facesFolder)

cap = cv2.VideoCapture(1)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

count = 0
while True:
    ret,frame = cap.read()
    print(frame)
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    k = cv2.waitKey(1)
    print (k)
    if k == 27:
        break

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(128,0,255),2)
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(facesFolder+'/rostro_{}.jpg'.format(count),rostro)
        cv2.imshow('rostro',rostro)
        count = count +1
    cv2.rectangle(frame,(10,5),(450,25),(255,255,255),-1)
    cv2.putText(frame,'Presione s, para almacenar los rostros encontrados',(10,20), 2, 0.5,(128,0,255),1,cv2.LINE_AA)
    cv2.imshow('frame',frame)

cap.release()
cv2.destroyAllWindows()
