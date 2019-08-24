# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 20:00:50 2019

@author: asd13
"""

import numpy as np
np.random.seed(1337)#確保隨機取出訓練數據的順序相同

from keras.datasets import mnist
(X_train,y_train),(X_test,y_test)=mnist.load_data()
print(X_train.shape)#(60000,28,28)
print(dir(X_train))
print(X_train.size)
print(X_train.shape)
print(X_train)

img_rows,img_cols=28,28
X_train=X_train.reshape(X_train.shape[0],img_rows,img_cols,1)#(60000,28,28,1)
X_test=X_test.reshape(X_test.shape[0],img_rows,img_cols,1)
input_shape=(img_rows,img_cols,1)#28,28,1

X_train=X_train.astype(np.float32)/255.0
X_test=X_test.astype(np.float32)/255.0
print(X_train)

#和前面依樣隊訓練標籤進行毒熱編碼 確保每個標籤對應一個輸出 可用sklearn preprocessing完成 但這裡用別ㄉ
from keras.utils import np_utils
n_classes=10
Y_train=np_utils.to_categorical(y_train,n_classes)
#print(Y_train)#60000*10
Y_test=np_utils.to_categorical(y_test,n_classes)

#創造一個convolutional neural network
from keras.models import Sequential
model=Sequential()

from keras.layers import Conv2D
n_filters=32
kernel_size=(3,3)
model.add(Conv2D(n_filters,(kernel_size[0],kernel_size[1]),padding='valid',input_shape=input_shape))

#linear rectified unit as an activation fun.
from keras.layers import Activation
model.add(Activation('relu'))

#again convolution
model.add(Conv2D(n_filters,(kernel_size[0],kernel_size[1])))
model.add(Activation('relu'))

#add dropout layer
from keras.layers import MaxPooling2D,Dropout
pool_size=(2,2)
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

#flat the model
from keras.layers import Flatten,Dense
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes))
model.add(Activation('softmax'))

#cross entropy loss and adadelta algorithm算法
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

#評分
model.fit(X_train,Y_train,batch_size=128,nb_epoch=12,verbose=1,validation_data=(X_test,Y_test))
#選擇要跌帶多少次 計算誤差梯度需要多少樣本(batch_size) 是否要對數據隨機化(shuffle) 是否輸出進度更新(verbose)
print(model.evaluate(X_test,Y_test,verbose=0))

#模型結構存檔
from keras.models import model_from_json
json_string=model.to_json() 
with open("model.condfig","w") as text_file:
    text_file.write(json_string)
model.save_weights("model.weight")
#儲存結構和銓重
from keras.models import load_model
model.save("model_33333.h5")