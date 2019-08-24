import numpy as np  
from keras.models import Sequential
from keras.models import model_from_json
from keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import cv2

"""
with open("model.config", "r") as text_file:
    json_string = text_file.read()
    model = Sequential()
    model = model_from_json(json_string)
    model.load_weights("model.weight", by_name=False)
# 刪除既有模型變數
del model 
"""
# 載入模型
model = load_model('model_33333.h5')
print('model load down')

#開鏡頭
cap = cv2.VideoCapture(0)
while(1):
    # get a frame
    ret, frame = cap.read()
    # show a frame
    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("./number.jpg", frame)
        break
cap.release()
cv2.destroyAllWindows()

#讀取圖片近來
im=cv2.imread('number.jpg')
#變成灰階
img=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
cv2.imwrite('number_gray.jpg',img)
#變成黑白
ret,imbw=cv2.threshold(img,125, 255, cv2.THRESH_BINARY_INV) #125
cv2.imwrite('number_bw.jpg',imbw)
#裁切成正方形 找一個通識
imcut=imbw[0:0+480,110:110+480]
cv2.imwrite('number_imcut.jpg',imcut)
#圖片銳化
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
imsharp = cv2.filter2D(imcut, -1, kernel=kernel)
cv2.imwrite('number_insharp.jpg',imsharp)
#降噪模糊
imblur=cv2.medianBlur(imsharp,3)
cv2.imwrite('number_imblur.jpg',imblur)
#膨脹
kernel=np.ones((13,13),np.uint8)
imfat=cv2.dilate(imblur,kernel,iterations=1)
cv2.imwrite('number_imfat.jpg',imfat)
#重新讀取
im=Image.open('number_imfat.jpg').convert('L')
im=im.resize((28,28),Image.ANTIALIAS)
imm=np.array(im)

#X_train=X_train.astype(np.float32)/255.0
X_train_2D = imm.reshape(1,28,28,1).astype('uint8')  
#print(X_train_2D)
x_Train_norm = X_train_2D.astype(np.float32)/255.0
print(x_Train_norm)
y=model.predict(x_Train_norm)
print(np.argmax(y))
print(y)
plt.imshow(im)
