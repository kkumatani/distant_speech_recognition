#  Module:  sfe.feature
#  Purpose: Audio Visual Speech recognition front end.
#  Author:  Munir Georges
#  Date: 2009
import string
from math import *

from opencv import cv
from opencv import highgui

from sfe.feature import*


def show(fr,width,height,name):
    image = cv.cvCreateImage(cv.cvSize (width, height),8,1)
    l = 0
    for j in range(0,image.width):
        for i in range(0,image.height):
            cv.cvSet2D(image,i,j,int(fr[l][0]));
            l=l+1
    highgui.cvShowImage(name,image)
    highgui.cvWaitKey(1000/29)




frame = VideoFeaturePtr(0,12)
frame.read('/projects/avicar/video/JF1/35D/JF1_35D.avi',0)
roi = ImageRoiPtr(frame,720,480,0,0,360,240)
fil = ImageFilterPtr(roi,360,240,3,5,5)
#hea = ImageDetectionPtr(fil,360,240,140,160,'haarcascade_frontalface_alt.xml', 1.2,2,1,5,3)
mou = ImageDetectionPtr(fil,360,240,100,50,'mouth25by15.xml', 1.8,2,1,60,30)
#lin = LinearInterpolationPtr(det,100,50,29,100)

#s = SaveImagePtr(100,50,'/home/mgeorges/test/')

window_name = "head"
highgui.cvNamedWindow(window_name,1)

 
frameX = 0 
for img in mou:
    #s.save(img,'test%04d.png' %frameX)
    for k in range(100):    
        fr = mou.next()
        show(fr,100,50,window_name)
    frameX += 1
    if frameX>9:
        break
    
print 'done with accumulating'

print 'finish'



#SaveImagePtr(width,height,directory)

#VideoFeature(mode,size) // mode = 0 <=> Gray-Values and size is not used!

#ImageRoiPtr(src,width,height,x1,y1,x2,y2)

#LinearInterpolationPtr(src,width,height,fps_from,fps_to)

#ImageShow(src,width,height) // not supportet by this python/swig/linking version!!

#ImageDetectionPtr(src,width,height,extraction_width,extraction_height,detection_stuff)
#detection_stuff:  filename of training, scale_factor,  min_neighbors, flags,  min_sizeX,  min_sizeY

#ImageFilterPtr(src,width,height,x,param1,param2)
# x = 0 = CV_BLUR_NO_SCALE (simple blur with no scaling)
# x = 1 = CV_BLUR (simple blur)
# x = 2 = CV_GAUSSIAN (gaussian blur)
# x = 3 = CV_MEDIAN (median blur)
# x = 4 = CV_BILATERAL (bilateral filter)
# // see the cv.h file in /opencv/include/opencv/
