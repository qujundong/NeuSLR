import tkinter as tk
import cv2
import sklearn
import numpy as np
import os
import tkinter as tk
from sklearn.neighbors import KNeighborsClassifier
index = 0
index_to_class={0:'start',1:'I am',2:'student',3:'end'}
train_path = 'train'
def get_KNN(train_path,index_to_class):
    neigh = KNeighborsClassifier(n_neighbors=3)
    imagefiles = os.listdir(train_path)
    train_X = []
    train_Y = []
    for imagefile in imagefiles:
        images = os.listdir(os.path.join(train_path,imagefile))
        for image in images:
            print (os.path.join(train_path,imagefile,image))
            img = cv2.imread(os.path.join(train_path,imagefile,image))
            #img = cv2.resize(img,(50,50),
            train_X.append(img.flatten())
            train_Y.append(int(imagefile))
    print (train_X)
    neigh.fit(train_X,train_Y)
    return neigh
def add_class(train_path,index_to_class): 
    global index
    var = e.get()
    classfile = os.path.join(train_path,str(index))
    if not os.path.exists(classfile):
            os.mkdir(classfile)
            index_to_class[index] = var
            index += 1
    cap = cv2.VideoCapture(0)
    i = 0
    while True :
        ret,frame = cap.read()
        frame = cv2.flip(frame,1)
        cv2.imshow('frame',frame)
        if i < 30:
            frame = cv2.resize(frame,(100,100),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(classfile+'/'+str(i)+'.jpg',frame)
        key = cv2.waitKey(100)
        if key == 27:
            break
        i += 1
    cap.release()
    cv2.destroyAllWindows()
    print(var)
def train_and_predict(train_path,index_to_class):
    knn = get_KNN(train_path,index_to_class)
    print('anything is ok')
    cap = cv2.VideoCapture(0)
    result = []
    while True:
        ret,frame  =cap.read()
        frame = cv2.flip(frame,1)
        cv2.imshow('frame',frame)
        frame = cv2.resize(frame,(100,100),interpolation=cv2.INTER_CUBIC)
        frame = frame.flatten()
        test = [frame]
        y_pred = knn.predict(test)
        y_prob = knn.predict_proba(test)
        print(y_pred,y_prob,index_to_class)
        if 'start' not in result:
            if y_pred[0] == 0 and y_prob[y_pred[0]] >= 0.8:
                result.append('start')
        else:
            if y_prob[y_pred[0]]>= 0.8 and index_to_class[y_pred[0]] == 'end':
                print(result)
                result =[]
            else:
                if y_prob[y_pred[0]] >= 0.8 and index_to_class[y_pred[0]] != result[-1] :
                    result.append(index_to_class[y_pred[0]])
        key = cv2.waitKey(100)
        if key == 27:
            
            break
    cap.release()
    cv2.destroyAllWindows()


window = tk.Tk()
window.title('My Window')
window.geometry('500x300')  
e = tk.Entry(window, show = None)
e.pack()

# 第6步，创建并放置两个按钮分别触发两种情况
b1 = tk.Button(window, text='add class', width=10,
               height=2, command=lambda:add_class(train_path = train_path,index_to_class=index_to_class))
b1.pack()
b2 = tk.Button(window, text='train and predict', width=10,
               height=2, command=lambda :train_and_predict(train_path=train_path,index_to_class=index_to_class))
b2.pack()
 
# 第8步，主窗口循环显示
window.mainloop()