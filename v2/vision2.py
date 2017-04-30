import pygame
from pygame.locals import *
import cv2,os,sys,time
import numpy as np

frontal_face_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(sys.argv[0]),"cascades/frontalFaceCascade.xml"))
eyes_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(sys.argv[0]),"cascades/haarcascade_eye.xml"))
full_body_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(sys.argv[0]),"cascades/haarcascade_fullbody.xml"))

video_capture = cv2.VideoCapture(0)

pygame.init()
pygame.display.set_caption("vision2")
win_width = 1280
win_height = 720
screen = pygame.display.set_mode([win_width,win_height])


video_capture = cv2.VideoCapture(0)

video_capture.set(3,1280)
video_capture.set(4,720)
time.sleep(0.5)
# video_capture.set(11,100)
# video_capture.set(15, -8.0)

target_square = pygame.image.load(os.path.join(os.path.dirname(sys.argv[0]),"data/target.png"))
while True:
    ret, image = video_capture.read()

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    screen.fill([0,0,0])

    faces_front = frontal_face_cascade.detectMultiScale(gray_image,1.3,5)
##    eyes = eyes_cascade.detectMultiScale(gray_image,1.3,6)
##    full_bodies = full_body_cascade.detectMultiScale(gray_image,1.2,2)
##    print(full_bodies)

##    for (x,y,width,height) in faces_front:
##        cv2.rectangle(image, (x,y), (x+width, y+height), (0,0,255),3)
##        region_of_interest_gray = gray_image[y:y+height, x:x+width]
##        region_of_interest_color = image[y:y+height, x:x+width]
##
##        for (eye_x,eye_y,eye_width,eye_height) in eyes:
##            cv2.rectangle(image, (eye_x,eye_y), (eye_x+eye_width, eye_y+eye_height), (0,255,255),3)
##            region_of_interest_gray = gray_image[eye_y:eye_y+eye_height, eye_x:eye_x+eye_width]
##            region_of_interest_color = image[eye_y:eye_y+eye_height, eye_x:eye_x+eye_width]
##
##    for (x,y,width,height) in full_bodies:
##        cv2.rectangle(image, (x,y), (x+width, y+height), (255,0,0),3)
##        region_of_interest_gray = gray_image[y:y+height, x:x+width]
##        region_of_interest_color = image[y:y+height, x:x+width]

    # b,g,r = cv2.split(image)
    # image = cv2.merge((b,g,r)
    display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    display_image = np.rot90(display_image)
    display_image = pygame.surfarray.make_surface(display_image)
    screen.blit(display_image,(0,0))

    for (x,y,width,height) in faces_front:
        screen.blit(target_square,(win_width-x-width,y))

    pygame.display.update()

    for event in pygame.event.get():
        if event.type == KEYDOWN:
            sys.exit(0)

video_capture.release()
