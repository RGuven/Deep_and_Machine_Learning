# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 09:10:37 2019

@author: Administrator
"""

import cv2

cap=cv2.VideoCapture(0)

while True:
    _,frame=cap.read()
    cv2.imshow("capture",frame)
    
    
    if cv2.waitKey(2) & 0xFF==ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()