#! C:\Python34\python3.exe
import cv2,os,sys
import numpy as np

# load cascade by creating an OpenCV Cascade Classifier, classifies a cascade

frontal_face_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(sys.argv[0]),"cascades/frontalFaceCascade.xml"))
eyes_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(sys.argv[0]),"cascades/haarcascade_eye.xml"))
full_body_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(sys.argv[0]),"cascades/haarcascade_fullbody.xml"))

# loads video camera labeled 0 by os
video_capture = cv2.VideoCapture(0)

while True:
    # captures image from loaded camera video_capture
    ret, image = video_capture.read()

    # converts that image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # lists faces detected by use of the cascade on the grayscale image. lists in the form of rectangles, (pos x, pos y, width ,height)
    faces_front = frontal_face_cascade.detectMultiScale(gray_image,1.3,10)
    eyes = eyes_cascade.detectMultiScale(gray_image,1.2,5)
    full_bodies = full_body_cascade.detectMultiScale(gray_image,1.2,5)

    for (x,y,width,height) in faces_front:
        cv2.rectangle(image, (x,y), (x+width, y+height), (0,0,255),3)
        region_of_interest_gray = gray_image[y:y+height, x:x+width]
        region_of_interest_color = image[y:y+height, x:x+width]

        for (eye_x,eye_y,eye_width,eye_height) in eyes:
            cv2.rectangle(image, (eye_x,eye_y), (eye_x+eye_width, eye_y+eye_height), (0,255,255),3)
            region_of_interest_gray = gray_image[eye_y:eye_y+eye_height, eye_x:eye_x+eye_width]
            region_of_interest_color = image[eye_y:eye_y+eye_height, eye_x:eye_x+eye_width]

    for (x,y,width,height) in full_bodies:
        cv2.rectangle(image, (x,y), (x+width, y+height), (255,0,0),3)
        region_of_interest_gray = gray_image[y:y+height, x:x+width]
        region_of_interest_color = image[y:y+height, x:x+width]

    cv2.imshow("image",image)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

video_capture.release()
cv2.destroyAllWindows()
