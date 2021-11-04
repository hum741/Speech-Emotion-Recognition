#!/usr/bin/env python
# -*- conding:utf-8 -*-
# Author: hum741
# CreatTime: 2021/10/17
# FileName: Backend2
# Description:对后端模型的融合方式进行测试

from keras.models import Sequential
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
import os #文件处理模块
import keras
import pickle #读取、存储文件
import tensorflow as tf
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

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
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
def focal_loss(y_true, y_pred):
    gamma = 4.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
# 读取特征
data = load_f(r'deep_feature_map1.pkl')#属性为list
labels = load_f(r'label.pkl')
data = Normalization(data)
x_train, y_train, x_val, y_val,x_test,y_test = Train_Val_Split(data,labels)
#划分训练集和测试集
# y_train = smooth_labels(y_train)
#**************将loss曲线左移半个epoch的模块
num_epochs = 500
# epochs = []
# for i in range(num_epochs):
#     epochs.append(i)
# epochs = np.array(epochs)

#*****************************模型训练*****************************

print(np.shape(x_train))
print(np.shape(x_val))
print(np.shape(y_train))
print(np.shape(y_val))
print(np.shape(x_test))

x_train = x_train.reshape(960,3072)
x_val = x_val.reshape(120,3072)
#**********定义模型**********
model = models.Sequential()#构建sequential顺序模型

model.add(layers.Dense(1024, activation='relu', input_shape=(3072,)))
model.add(layers.Dropout(0.1))   # 0.2是神经元舍弃比
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.2))
# model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(6, activation = 'softmax'))
plot_model(model, to_file='end_model.png', show_shapes=True)#做出2D_CNN结构图
model.summary()

opt = optimizers.Adam(lr=0.00001, decay=1e-6)#定义RMS优化器，用于优化损失函数
# model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy',f1_m,precision_m, recall_m])#定义损失函数ADAMval
model.compile(loss=[focal_loss], optimizer=opt,metrics=['accuracy'])
callbacks_list = [keras.callbacks.EarlyStopping(monitor='accuracy',patience=50,verbose=0,mode='auto')]#提前停止
keras.callbacks.ModelCheckpoint(filepath='speechmfcc_model_checkpoint.h5',monitor='val_loss',save_best_only=True)#该回调函数将在每个epoch后保存模型到filepath

history = model.fit(x_train, y_train,epochs=num_epochs,batch_size=128,validation_data=(x_val, y_val),shuffle=True,callbacks=callbacks_list)

model.save('speech_mfcc_model4.h5')
model.save_weights('speech_mfcc_model_weight4.h5')
#可视化训练结果
# fig = plt.figure(figsize=(10,4))
# ax1 = fig.add_subplot(1,2,1)
# ax2 = fig.add_subplot(1,2,2)
#
# ax1.plot(epochs-0.5,history.history['loss'])
# ax1.plot(epochs,history.history['val_loss'])
# ax1.set_title('model loss')
# ax1.set_ylabel('loss')
# ax1.set_xlabel('epoch')
# ax1.legend(['train', 'test'], loc='upper left')
#
# ax2.plot(epochs-0.5,history.history['accuracy'])
# ax2.plot(epochs,history.history['val_accuracy'])
# ax2.set_title('mix model acc')
# ax2.set_ylabel('acc')
# ax2.set_xlabel('epoch')
# ax2.legend(['train', 'test'], loc='upper left')
# plt.show()

print(history.history['val_accuracy'])
print(history.history['accuracy'])
print(history.history['val_loss'])
print(history.history['loss'])


# evaluate the model
# loss, accuracy, f1_score, precision, recall = model.evaluate(x_test, y_test, verbose=0)
eval = model.evaluate(x_test,y_test,batch_size=120,verbose=0)
print('error=',eval[0])
print('accurcy=',eval[1])

fig, ax1 = plt.subplots()
ax1.plot(history.history['loss'],c='orangered')
ax2 = ax1.twinx()
ax2.plot(history.history['accuracy'],c='green')
ax2.plot(history.history['val_accuracy'],c='blue')
ax1.set_title('mix model')
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax2.set_ylabel('accuracy')
plt.legend(['train acc', 'test acc'], loc='best')
plt.show()