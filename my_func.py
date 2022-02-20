import cv2
import numpy as np
def personalNormBin(grey_image):
    ## normalize image
    original_as_float = grey_image.astype(np.float32) / 255
    original_as_float -= original_as_float.mean()
    original_as_float /= original_as_float.std()
    
    normalized_final = np.sign(original_as_float)
    normalized_final = normalized_final.astype(np.uint8)
    inverted = cv2.bitwise_not(normalized_final)

    return inverted

def haar_classifier(grey_image):
    
    found = False
    images=[]
    kernels = [51,61,71, 101,121,131]
    for ks in kernels:
        images.append(cv2.GaussianBlur(grey_image,(ks,ks),0))
    
    left_ear_cascade = cv2.CascadeClassifier('.\\haarcascade_mcs_leftear.xml')
    right_ear_cascade = cv2.CascadeClassifier('.\\haarcascade_mcs_rightear.xml')
    
    left_ear = []
    for min_neighbors in range(6,0,-1):#6,5,4,3,2,1
        for image in images:
            resL = left_ear_cascade.detectMultiScale(image,1.1,min_neighbors)
            resR = right_ear_cascade.detectMultiScale(image,1.1,min_neighbors)
            if type(resL)!=type(()):
                found = True
                left_ear.append(resL[0])
            if type(resR)!=type(()):
                found = True
                left_ear.append(resR[0])

    
    #img = cv2.cvtColor(grey_image,cv2.COLOR_GRAY2RGB)

    min_x=grey_image.shape[1]
    min_y=grey_image.shape[0]
    max_x = 0
    max_y = 0
    for (x,y,w,h) in left_ear:
        if x<min_x:
            min_x=x
        if y<min_y:
            min_y=y
        if (x+w)>max_x:
            max_x = x+w
        if (y+h)>max_y:
            max_y = y+h
        
    #cv2.rectangle(img, (min_x,min_y), (max_x,max_y), (0,255,0), 3)
        

    rectangle = (min_x,min_y, max_x-min_x, max_y-min_y)   
    if found:
        img = grey_image[min_y:max_y, min_x:max_x]
    else:
        img = grey_image
    return (img, found, rectangle)

