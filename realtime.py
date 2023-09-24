import cv2
import numpy as np
from keras.models import load_model
import time
import sys
import os

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model('emotion_recognition.h5')
cap = cv2.VideoCapture(0)

faceCascade = face_detector
font = cv2.FONT_HERSHEY_SIMPLEX


emotions = {0:'Angry',1:'Fear',2:'Happy',3:'Sad',4:'Surprised',5:'Neutral'}





frame_count = 0

while True:
        
        ret, frame = cap.read()

        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        
        start_time = time.time()
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(100, 100),)
        
        y0 = 15
        for index in range(6):
            cv2.putText(frame, emotions[index] + ': ', (5, y0), font,
                        0.4, (255, 0, 255), 1, cv2.LINE_AA)
            y0 += 15

       
        FIRSTFACE = True
        if len(faces) > 0:
            for x, y, width, height in faces:
                cropped_face = gray[y:y + height,x:x + width]
                test_image = cv2.resize(cropped_face, (48, 48))
                test_image = test_image.reshape([-1,48,48,1])

                test_image = np.multiply(test_image, 1.0 / 255.0)

                # Probablities of all classes
                #Finding class probability takes approx 0.05 seconds
                start_time = time.time()
                if frame_count % 5 == 0:
                    probab = model.predict(test_image)[0] * 100
                    #print("--- %s seconds ---" % (time.time() - start_time))

                    #Finding label from probabilities
                    #Class having highest probability considered output label
                    label = np.argmax(probab)
                    probab_predicted = int(probab[label])
                    predicted_emotion = emotions[label]
                    frame_count = 0

                frame_count += 1
                font_size = width / 300
                filled_rect_ht = int(height / 5)
                #Drawing probability graph for first detected face
                if FIRSTFACE:
                    y0 = 8
                    for score in probab.astype('int'):
                        cv2.putText(frame, str(score) + '% ', (80 + score, y0 + 8),
                                    font, 0.3, (0, 0, 255),1, cv2.LINE_AA)
                        cv2.rectangle(frame, (75, y0), (75 + score, y0 + 8),
                                      (0, 255, 255), cv2.FILLED)
                        y0 += 15
                        FIRSTFACE =False

                
                #Drawing rectangle and showing output values on frame
                cv2.rectangle(frame, (x, y), (x + width, y + height),(155,155, 0),2)
                cv2.rectangle(frame, (x-1, y+height), (x+1 + width, y + height+filled_rect_ht),
                              (155, 155, 0),cv2.FILLED)
                cv2.putText(frame, predicted_emotion+' '+ str(probab_predicted)+'%',
                            (x, y + height+ filled_rect_ht-10), font,font_size,(255,255,255), 1, cv2.LINE_AA)

               

       

        cv2.imshow('frame', frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
out.release()
cap.release()
cv2.destroyAllWindows()


    
   
