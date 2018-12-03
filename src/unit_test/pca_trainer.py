
from sfe.feature import*
frame = VideoFeaturePtr(1,720,480)
frame.read('/projects/avicar/video/BF1/35D/BF1_35D.avi',0,200)
roi = ImageRoiPtr(frame,720,480,0,0,360,240)
#det = ImageDetectionPtr(roi,360,240,100,50,'mouth25by15.xml', 1.8,6,1,50,30)
pcaE = PCAModEstimatorPtr(10,360*240);
#pcaE = PCAModEstimatorPtr(10,50*100);
#i = ImageShowPtr(det,100,50)
s = SaveImagePtr(360,240,'/home/mgeorges/Desktop/')
#s = SaveImagePtr(100,50,'/home/mgeorges/Desktop/')

frameX = 0 
for image in roi:
    pcaE.accumulate(image)
    #s.save(image,'test%04d.png' %frameX)
    for j in range(100):
        roi.next()
    frameX += 1
    if frameX>9:
        break
print 'done with accumulating'

pcaE.estimate()
pcaE.save('/home/mgeorges/Desktop/test/test.bin','/home/mgeorges/Desktop/test/test2.bin');
s.savedouble(pcaE.get_mean(),'mean.png')
s.savedouble(pcaE.get_vec(0),'vec0.png')
s.savedouble(pcaE.get_vec(1),'vec1.png')
s.savedouble(pcaE.get_vec(2),'vec2.png')
s.savedouble(pcaE.get_vec(3),'vec3.png')
s.savedouble(pcaE.get_vec(4),'vec4.png')
s.savedouble(pcaE.get_vec(5),'vec5.png')
s.savedouble(pcaE.get_vec(6),'vec6.png')
s.savedouble(pcaE.get_vec(7),'vec7.png')
s.savedouble(pcaE.get_vec(8),'vec8.png')
s.savedouble(pcaE.get_vec(9),'vec9.png')


print 'finish'
    
# VideoFeature(mode,size) // mode = 0 <=> Gray-Values and size is not used!
# ImageRoiPtr(src,width,height,x1,y1,x2,y2)
# ImageDetectionPtr(src,width,height,extraction_width,extraction_height,detection_stuff)
# LinearInterpolationPtr(src,width,height,fps_from,fps_to)
# ImageShow(src,width,height)
