import numpy as np
import os
#from scipy.misc import imread,imresize
import PIL
from keras.models import model_from_json
import tensorflow as tf
import pickle
import cv2
dataColor = (0,255,0)
font = cv2.FONT_HERSHEY_SIMPLEX
fx, fy, fh = 10, 50, 45
takingData = 0
className = 'NONE'
count = 0
showMask = 0
# Loading int2word dict
classifier_f = open("int_to_word_out.pickle", "rb")
int_to_word_out = pickle.load(classifier_f)
classifier_f.close()


def load_model():
    
    # load json and create model
    json_file = open('model_face.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_face.h5")
    print("Loaded model from disk")
    return loaded_model

def pre_process(image):
    
    image = image.astype('float32')
    image = image / 255.0
    return image

def load_image():


    img=os.listdir("images")[0]
    image=np.array(cv2.imread("predict/"+img))
    image = cv2.imresize(image, (64, 64))
    image=np.array([image])
    image=pre_process(image)
    return image
def load_video():
    model=load_model()
    cam=cv2.VideoCapture(0)
    ret,frame=cam.read()
    cv2.imwrite('test.jpg',frame)
    while 1:
        ret2,frame2=cam.read()
        x0, y0, width = 200, 220, 300
        cv2.rectangle(frame2, (x0,y0), (x0+width-1,y0+width-1),dataColor, 12)
     
        roi = frame2[y0:y0+width,x0:x0+width]
        cv2.imwrite('test.jpg',roi)
        image=cv2.imread('test.jpg')
        image = cv2.resize(image, (64, 64))
        image=np.array([image])
        image=pre_process(image)
        
        #frame2[y0:y0+width,x0:x0+width] = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        pred = model.predict(image)
        pred=int_to_word_out[np.argmax(pred)]
        cv2.putText(frame2, 'Prediction: %s' % (pred), (fx,fy+2*fh), font, 1.0, (245,210,65), 2, 1)
        cv2.imshow('Original', frame2)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cam.release()
            cv2.destroyAllWindows()
        
        
        
        
'''
def load_video():
    model = load_model()
    #model = tf.keras.models.load_model('model_face.h5')
    
    x0, y0, width = 200, 220, 300
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cam = cv2.VideoCapture(0)
    while i>1:
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1) # mirror
        window =frame
        cv2.rectangle(window, (x0,y0), (x0+width-1,y0+width-1),dataColor, 12)
        roi = frame[y0:y0+width,x0:x0+width]
        img = np.float32(roi)/255.
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)
        pred = classes[np.argmax(model.predict(img))]
        cv2.putText(window, 'Prediction: %s' % (pred), (fx,fy+2*fh), font, 1.0, (245,210,65), 2, 1)
        cv2.imshow('Original', window)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cam.release()
        cv2.destroyAllWindows()
'''
    
        
'''
image=load_image()
model=load_model()
prediction=model.predict(image)

print(prediction)
print(np.max(prediction))
print(int_to_word_out[np.argmax(prediction)])
'''
load_video()
