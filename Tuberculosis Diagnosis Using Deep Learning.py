#!/usr/bin/env python
# coding: utf-8

# In[59]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D


# In[61]:


train_datadir="D:/Fyp/ChinaSet_AllFiles/ChinaSet_AllFiles/fypDataset/training"
categories=["negative","positive"]
img_size=100


# In[62]:


training_data=[]
def training_model_data():
    
    for category in categories:
        path=os.path.join(train_datadir,category) # it is use for the path of negative and positive 
        class_num=categories.index(category) # it will be for 0 (for negative) or 1 (for positive)
        for img in os.listdir(path):
            try:
                img_array= cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE) 
                # if we dont change image into grayscale then the data will be in 3D
                new_array= cv2.resize(img_array,(img_size,img_size))
                training_data.append([new_array,class_num])
            except Exception as e:
                print(e)
                break
                
training_model_data()


# In[63]:


random.shuffle(training_data)


# In[64]:


for sample in training_data[:20]:
    print(sample[1])


# In[65]:


x=[]
y=[]
for features,labels in training_data:
    x.append(features)
    y.append(labels)

x=np.array(x).reshape(-1,img_size,img_size,1)
y=np.array(y)


# # train model on 3 hidden layers and batch_size=42

# In[66]:


x=x/255.0
model=Sequential()
model.add( Conv2D(64, (3,3) , input_shape=(100, 100, 1)) )
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3) ))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) #it will convert our features data from 3d to 1d
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

hist=model.fit(x,y,batch_size=42,epochs=12,validation_split=0.1)


# In[71]:


accuracy=hist.history['accuracy']
val_acc=hist.history['val_accuracy']
loss=hist.history['loss']
val_loss=hist.history['val_loss']


# In[72]:


epochs=range(1,len(accuracy)+1)
plt.plot(epochs,accuracy,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training & Validation Accuracy')
plt.legend()
plt.figure()


# In[73]:


plt.plot(epochs,loss,'bo',color='black',label='Training Loss')
plt.plot(epochs,val_loss,'b',label='Validation Loss')
plt.title('Training & validation Loss')
plt.legend()
plt.figure()


# In[74]:


plt.plot(epochs,accuracy,color='blue',label='Training acc')
plt.plot(epochs,val_acc,color='green',label='Validation acc')
plt.plot(epochs,loss,color='red',label='Training Loss')
plt.plot(epochs,val_loss,color='black',label='Validation Loss')
plt.title('Traing & Validation with Loss & Accuracy')
plt.legend()


# In[182]:


def prepare(filepath):
   # print(filepath)
    img_size=100
    img_array=cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    
    new_array=cv2.resize(img_array,(img_size,img_size))
    return new_array.reshape(-1,img_size,img_size,1)

prediction=model.predict([prepare("D:/Fyp/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png/testing/positive/CHNCXR_0330_1.png")])
#prediction=model.predict([prepare("D:/Fyp/images/images/TEST_px29.jpg")])
print(categories[int(prediction)])


# In[73]:


path='D:/Fyp/ChinaSet_AllFiles/ChinaSet_AllFiles/fypDataset/testing/positive'
#category=['negative','positive']
def prepare(filepath):
    #print(filepath)
    img_size=100
    img_array=cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    new_array=cv2.resize(img_array,(img_size,img_size))
    plt.imshow(new_array,cmap=plt.get_cmap("gray"))
    plt.show()
    return new_array.reshape(-1,img_size,img_size,1)


for img in os.listdir(path): 
        try:
            #print(img)
            prediction=model.predict([prepare(path+'/'+img)])
            print(categories[int(prediction)])
        except IOError: 
             pass


# # testing Model 2

# In[29]:


from tqdm import tqdm
import os
from random import shuffle
import numpy as np
import cv2
TEST_DIR='D:/Fyp/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png/testing3'
IMG_SIZE=250
def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('_')[2]
        img_num=img_num.split('.')[0]
        print(img_num)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data
#process_test_data()
test_data=process_test_data()


# In[33]:


for sample in test_data:
    print(sample[1])


# In[52]:


import matplotlib.pyplot as plt

fig=plt.figure(figsize=(100,100))
for num,data in enumerate(test_data[:10]):    
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(5,2,num+1)
    orig = img_data
    data = img_data.reshape(-1,250,250,1)
    data=np.array(data)
  #  print(type(data))
 #   print("________________________")
#    print(data)
    #model.predict([data])[0]
    #model.predict(data)
    model_out = model.predict([data])[0]
   # print(model_out)
    if np.argmax(model_out) == 1: 
        str_label='positive' 
        print(np.argmax(model_out))
    else: 
        str_label='Negative'
    print(np.argmax(model_out))
        
    y.imshow(orig,cmap='gray')
    plt.title(str_label, fontsize=100)
    #y.axes.get_xaxis().set_visible(False)
    #y.axes.get_yaxis().set_visible(False)
plt.show()


# In[ ]:




x=x/255.0
model=Sequential()
model.add( Conv2D(32, (3,3) , input_shape=(100, 100, 1)) )
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3) ))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3) ))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) #it will convert our features data from 2d to 1d
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

hist=model.fit(x,y,batch_size=36,epochs=14,validation_split=0.1)x=x/255.0
model=Sequential()
model.add( Conv2D(32, (3,3) , input_shape=(100, 100, 1)) )
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3) ))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3) ))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) #it will convert our features data from 2d to 1d
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

hist=model.fit(x,y,batch_size=42,epochs=12,validation_split=0.1)x=x/255.0
model=Sequential()
model.add( Conv2D(32, (3,3) , input_shape=x.shape[1:]) )
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3) ))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3) ))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) #it will convert our features data from 2d to 1d
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

hist=model.fit(x,y,batch_size=42,epochs=12,validation_split=0.1)x=x/255.0
model=Sequential()
model.add( Conv2D(32, (1,1),strides=(1,1),padding='same' , input_shape=x.shape[1:]) )
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(1,1),strides=(1,1) ))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(1,1),strides=(2,2) ))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) #it will convert our features data from 2d to 1d
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

hist=model.fit(x,y,batch_size=28,epochs=18,validation_split=0.1)