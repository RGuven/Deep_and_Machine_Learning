# -*- coding: utf-8 -*-
"""


@author: Administrator
"""

import numpy as np
from PIL import ImageGrab
import cv2
import time

tik=time.time()

while(1):
    screen=np.array(ImageGrab.grab(bbox=(0,5,480,640)))
    tok=time.time()
    cv2.imshow("Window",cv2.cvtColor(screen,cv2.COLOR_BGR2RGB))
    print((tok-tik)*1000)
    if cv2.waitKey(2) & 0xFF==ord("q"):
        cv2.destroyAllWindows()
        break
    