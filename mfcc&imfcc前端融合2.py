#!/usr/bin/env python
# -*- conding:utf-8 -*-
# Author: hum741
# CreatTime: 2021/10/30
# FileName: mfcc&imfcc前端融合2
# Description:采用通道叠加的方式


import numpy as np
import os #文件处理模块
import keras
import pickle #读取、存储文件
from keras import backend as K
from keras import layers#神经网络的基本模块
from keras import models#模型构建：线性模型，功能定制接口
from keras import optimizers#优化器，包含多种优化算法
from keras.utils import plot_model#神经网络模型可视化
from keras import regularizers#正则化
from keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
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
    # x_train = np.expand_dims(x_train,axis=data.ndim)#将数据集扩充维度，输入网络
    # x_val = np.expand_dims(x_val,axis=data.ndim)
    # x_test = np.expand_dims(x_test,axis=data.ndim)
    return x_train, y_train, x_val, y_val,x_test,y_test
def smooth_labels(labels, factor=0.01):#标签平滑
    # smooth the labels
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    # returned the smoothed labels
    return labels
def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
#调用函数读取数据
mfcc_imfcc0 = load_f(r'mfcc_imfcc_0.pkl')#纵向拼接的特征
mfcc_imfcc1 = load_f(r'mfcc_imfcc_1.pkl')#通道叠加的特征
labels = load_f(r'label.pkl')
mfcc_imfcc0 = Normalization(mfcc_imfcc0)
mfcc_imfcc1 = Normalization(mfcc_imfcc1)
#分割训练集测试集
x_train, y_train, x_val, y_val,x_test,y_test = Train_Val_Split(mfcc_imfcc1,labels)
y_train = smooth_labels(y_train)
#*****************************模型训练*****************************
#**********定义模型**********
model = models.Sequential()#构建sequential顺序模型
#第一层
model.add(layers.Conv2D(input_shape=(39,126,2),filters=32, kernel_size=(3,3),strides=2, padding='valid',activation='relu',data_format='channels_last',name='CONV1'))

model.add(layers.Dropout(0.24,name='DP1'))
model.add(layers.Conv2D(filters=64, kernel_size=(3,3), strides=2, padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.002),name='CONV2'))
model.add(layers.Dropout(0.24,name='DP2'))#防止过拟合，随机去掉20%的神经元连接
model.add(layers.Conv2D(filters=128, kernel_size=3, strides=2,activation='relu', padding='valid', kernel_regularizer=regularizers.l2(0.002),name='CONV3'))
model.add(layers.Dropout(0.24,name='DP3'))#防止过拟合，随机去掉20%的神经元连接
model.add(layers.Conv2D(filters=256, kernel_size=3, strides=2, activation='relu', padding='valid',kernel_regularizer=regularizers.l2(0.002),name='CONV4'))
model.add(layers.Dropout(0.24,name='DP4'))#防止过拟合，随机去掉20%的神经元连接
model.add(layers.Flatten(name='FLT'))#将多维变量变为二维变量，因全连接层的输入只能为二维变量
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.1))   # 0.2是神经元舍弃比
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.2))
# model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(6, activation = 'softmax'))
plot_model(model, to_file='mfcc_model.png', show_shapes=True)#做出2D_CNN结构图
model.summary()
#**********编译模型**********
#配置学习过程 （多分类问题）
opt = optimizers.RMSprop(lr=0.00001, decay=1e-6)#定义RMS优化器，用于优化损失函数
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])#定义损失函数ADAMval

callbacks_list = [keras.callbacks.EarlyStopping(monitor='accuracy',patience=50,verbose=0,mode='auto')]#提前停止
keras.callbacks.ModelCheckpoint(filepath='speechmfcc_model_checkpoint.h5',monitor='val_loss',save_best_only=True)#该回调函数将在每个epoch后保存模型到filepath

#history = model.fit(x_train, y_train, epochs=50,batch_size=16,validation_data=(x_val, y_val))#模型训练
history = model.fit(x_train, y_train,epochs=1000,batch_size=64,validation_data=(x_val, y_val),shuffle=True,callbacks=callbacks_list)

model.save('mfcc_model.h5')
#model.save_weights('mfcc_model_weight.h5')

eval = model.evaluate(x_test,y_test,batch_size=120,verbose=0)
print('error=',eval[0])
print('accurcy=',eval[1])

fig, ax1 = plt.subplots()
ax1.plot(history.history['loss'],c='orangered')
ax2 = ax1.twinx()
ax2.plot(history.history['accuracy'],c='green')
ax2.plot(history.history['val_accuracy'],c='blue')
ax1.set_title('mfcc imfcc stack model')
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax2.set_ylabel('accuracy')
plt.legend(['train acc', 'test acc'], loc='lower right')
plt.show()



