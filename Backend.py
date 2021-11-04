#!/usr/bin/env python
# -*- conding:utf-8 -*-
# Author: hum741
# CreatTime: 2021/10/6
# FileName: Backend
# Description：后端分类器训练
#!/usr/bin/env python
#-*- conding:utf-8 -*-
# Author: hum741
# CreatTime: 2021/7/14
# FileName: CNN_network2.py
# Description: 通过DNN作为分类器进行模型融合

from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import os #文件处理模块
import keras
import pickle #读取、存储文件
from keras import layers#神经网络的基本模块
from keras import models#模型构建：线性模型，功能定制接口
from keras import optimizers#优化器，包含多种优化算法
from keras.utils import plot_model#神经网络模型可视化
from keras.optimizers import Adam
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
def load_f(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        file = np.array(file)#转化为矩阵格式
    return file
def Normalization(Input):
    DATA_MEAN = np.mean(Input, axis=0)  # 压缩行，对列求平均值，返回一个1*126*126*4的矩阵
    DATA_STD = np.std(Input, axis=0)  # 计算每一列的标准差,返回一个1*126*126*4的矩阵
    Output = Input - DATA_MEAN  # 减去平均值
    Output = Output/DATA_STD  # 除以标准差，数据标准化处理
    return Output
def Train_Val_Split(data,label):
    ratioTrain = 0.8
    ratio_test_val = 0.9
    numTrain = int(data.shape[0] * ratioTrain)  # 训练集的数目=数据条数*训练比
    numVal = int(data.shape[0] * ratio_test_val)-numTrain#验证集的数目
    permutation = np.random.permutation(data.shape[0])  # 生成1200个随机数，将全部数据打乱，说话者，情感种类全部打乱
    data = data[permutation, :]  # 将数据按照随机数的顺序重新排列
    Label = label[permutation, :]  # 将标记按照随机数的顺序重新排列
    x_train = data[:numTrain]  # 训练集
    x_val = data[numTrain:numTrain+numVal]  # 验证集
    x_test = data[numTrain+numVal:]
    y_train = Label[:numTrain]  # 训练集的label
    y_val = Label[numTrain:numTrain+numVal]  # 验证集的label
    y_test = Label[numTrain+numVal:]
    x_train = np.expand_dims(x_train,axis=data.ndim)#将数据集扩充维度，输入网络
    x_val = np.expand_dims(x_val,axis=data.ndim)
    # x_test = np.expand_dims(x_test,axis=data.ndim)
    return x_train, y_train, x_val, y_val,x_test,y_test
def smooth_labels(labels, factor=0.1):#标签平滑
    # smooth the labels
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    # returned the smoothed labels
    return labels
# 读取特征
data = load_f(r'deep_feature_map2.pkl')#属性为list
labels = load_f(r'label.pkl')
x_train, y_train, x_val, y_val,x_test,y_test = Train_Val_Split(data,labels)
y_train = smooth_labels(y_train)#仅对训练集进行标签平滑
#划分训练集和测试集

#*****************************模型训练*****************************

print(np.shape(x_train))
print(np.shape(x_val))
print(np.shape(y_train))
print(np.shape(y_val))

x_train = x_train.reshape(960,24576)
x_val = x_val.reshape(120,24576)
#**********定义模型**********5
model = models.Sequential()#构建sequential顺序模型

model.add(layers.Dense(512, activation='relu', input_shape=(24576,)))
model.add(layers.Dropout(0.1))   # 0.2是神经元舍弃比
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dropout(0.5))
model.add(layers.Dense(6, activation = 'softmax'))
plot_model(model, to_file='end_model.png', show_shapes=True)#做出2D_CNN结构图
model.summary()


opt = optimizers.RMSprop(lr=0.0001, decay=1e-6)#定义RMS优化器，用于优化损失函数
model.compile(loss='categorical_crossentropy', optimizer=Adam(),metrics=['accuracy'])#定义损失函数ADAMval

callbacks_list = [keras.callbacks.EarlyStopping(monitor='accuracy',patience=50,verbose=0,mode='auto')]#提前停止
keras.callbacks.ModelCheckpoint(filepath='speechmfcc_model_checkpoint.h5',monitor='val_loss',save_best_only=True)#该回调函数将在每个epoch后保存模型到filepath

#history = model.fit(x_train, y_train, epochs=50,batch_size=16,validation_data=(x_val, y_val))#模型训练
history = model.fit(x_train, y_train,epochs=300,batch_size=64,validation_data=(x_val, y_val),shuffle=True,callbacks=callbacks_list)

model.save('speech_mfcc_model4.h5')
model.save_weights('speech_mfcc_model_weight4.h5')

fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)


#可视化训练结果
ax1.plot(history.history['loss'])
ax1.plot(history.history['val_loss'])
ax1.set_title('model loss')
ax1.set_ylabel('loss')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'test'], loc='upper left')
# ax1.show()


ax2.plot(history.history['accuracy'])
ax2.plot(history.history['val_accuracy'])
ax2.set_title('mix model acc')
ax2.set_ylabel('acc')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'test'], loc='upper left')
plt.show()

print(history.history['val_accuracy'])
print(history.history['accuracy'])
print(history.history['val_loss'])
print(history.history['loss'])

eval = model.evaluate(x_test,y_test,batch_size=120,verbose=0)
print('error=',eval[0])
print('accurcy=',eval[1])
