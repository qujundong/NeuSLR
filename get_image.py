#-*- coding: utf-8 -*- 
import copy
import tkinter as tk
import cv2
import sklearn
import numpy as np
import os
import threading
import tkinter as tk
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image, ImageTk
import shutil
index = 0
index_to_class={}
class_to_index={}
train_path = 'train'
cap = cv2.VideoCapture(0)
with open('index_to_class.txt','r') as f:
    lines = f.readlines()
    if '' in lines:
        lines.remove('')
    print(lines,index)
    if '\n' in lines:
        lines.remove('\n')
    index = len(lines)
    for i,value in enumerate(lines) :
        value = value.strip('\n')
        index_to_class[i] = value
        class_to_index[value] = i
    print(index_to_class)
 #get the KNN classify
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
#open the videocapture
def open_video():
    def cc():
        global cap
        while True:
            ret, frame = cap.read()#从摄像头读取照片
            frame = cv2.flip(frame, 1)#翻转 0:上下颠倒 大于0水平颠倒
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            image_file=ImageTk.PhotoImage(img)
            canvas.create_image(0,0,anchor='nw',image=image_file)
            '''cv2.imshow('frame',frame)
            key = cv2.waitKey(100)
            if key == 27:
                break'''
        
    t=threading.Thread(target=cc)
    t.start()
#add class 
def add_class(train_path,index_to_class): 
    global index
    global class_to_index
    var = e.get()
    image_num = 0
    print(class_to_index)
    if var not in(class_to_index.keys()):
        classfile = os.path.join(train_path,str(index))
    else:
        file_index = class_to_index[var]
        classfile = os.path.join(train_path,str(file_index))
        image_num = len(os.listdir(classfile))
    if not os.path.exists(classfile):
            os.mkdir(classfile)
            index_to_class[index] = var
            class_to_index[var] = index
            index += 1
    global cap
    i = 0
    for i in range(5) :
        ret,frame = cap.read()
        frame = cv2.flip(frame,1)
        frame = cv2.resize(frame,(100,100),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(classfile+'/'+str(image_num+i)+'.jpg',frame)
            
    with open('index_to_class.txt','w') as f:
        values = ''
        for i in index_to_class.keys():
            values = values + index_to_class[i]+'\n'
        values.strip('\n')
        f.write(values)
    print(var)
'''   if index not in index_to_class.key():
        index += 1'''
#train the knn and get the prediction
def train_and_predict(train_path,index_to_class):
    def cc():
        flag = 0
        knn = get_KNN(train_path,index_to_class)
        print('anything is ok')
        result = []
        global cap
        while True:
            key = 0
            ret,frame  =cap.read()
            frame = cv2.flip(frame,1)
            img = copy.deepcopy(frame)
            string = ''
            for j in result:
                    string +=' ' + j
            if flag % 3 ==0:
                
                #cv2.imshow('frame',frame)
                frame = cv2.resize(frame,(100,100),interpolation=cv2.INTER_CUBIC)
                frame = frame.flatten()
                test = [frame]
                y_pred = knn.predict(test)
                #print(type(y_pred))
                y_prob = knn.predict_proba(test)
                print(y_pred[0],y_prob,index_to_class[y_pred[0]])
                
                #cv2.putText(img, string , (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (155,255,255), 2)
                if 'start' not in result:
                    if y_pred[0] == 0 and y_prob[0,y_pred[0]] >= 0.8:
                        result.append('start')
                else:
                    if y_prob[0,y_pred[0]]>= 0.8 and index_to_class[y_pred[0]] == 'end':
                        print(result)
                        result =[]
                    else:
                        if y_prob[0,y_pred[0]] >= 0.8 and y_pred[0]!=0 and  index_to_class[y_pred[0]] != result[-1] :
                            result.append(index_to_class[y_pred[0]])
            cv2.putText(img, string , (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (155,255,255), 2)
            cv2.imshow('Image',img)
            key = cv2.waitKey(3)
            if key ==27:
                break
            flag += 1
        #cap.release()
        cv2.destroyAllWindows()
    t=threading.Thread(target=cc)
    t.start()
#delete all the images and the names
def clear_image(train_path):
    classpaths = os.listdir(train_path)
    for file in classpaths:
        shutil.rmtree(os.path.join(train_path,file))
    with open('index_to_class.txt','w') as f:
        f.truncate() 
    global index 
    index= 0
    global index_to_class
    global class_to_index 
    index_to_class = {}
    class_to_index={}
def get():
    for i in index_to_class.values():
        print(i)
window = tk.Tk()
window.title('my window')
sw = window.winfo_screenwidth()#获取屏幕宽
sh = window.winfo_screenheight()#获取屏幕高
wx = 600
wh = 800
window.geometry("%dx%d+%d+%d" %(wx,wh,(sw-wx)/2,(sh-wh)/2-100))#窗口至指定位置
canvas = tk.Canvas(window,bg='#c4c2c2',height=wh,width=wx)#绘制画布
canvas.pack()


e = tk.Entry(window, show = None)
e.place(x=200,y=500)
b0 = tk.Button(window,text='open video',width=10,height=1,command=open_video)
b0.place(x=230,y=530)
b1 = tk.Button(window, text='add class', width=10,
               height=1, command=lambda:add_class(train_path = train_path,index_to_class=index_to_class))
b1.place(x=230,y=560)
b2 = tk.Button(window, text='train and predict', width=10,
               height=1, command=lambda :train_and_predict(train_path=train_path,index_to_class=index_to_class))
b2.place(x=230,y=590)
b3 = tk.Button(window,text='clear images',width=10,height=1,command=lambda:clear_image(train_path=train_path))
b3.place(x=230,y=620)
b3 = tk.Button(window,text='get classes',width=10,height=1,command=get)
b3.place(x=230,y=650)
window.mainloop()
