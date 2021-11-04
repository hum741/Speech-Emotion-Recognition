#!/usr/bin/env python
# -*- conding:utf-8 -*-
# Author: hum741
# CreatTime: 2021/10/4
# FileName: spectrum_training
# Description:基于spectrum图谱进行训练
import matplotlib.pyplot as plt
import numpy as np
import os #文件处理模块
import keras
import pickle #读取、存储文件
from keras import layers#神经网络的基本模块
from keras import models#模型构建：线性模型，功能定制接口
from keras import optimizers#优化器，包含多种优化算法
from keras.utils import plot_model#神经网络模型可视化
from keras import regularizers#正则化
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
    data = data[permutation, :, :]  # 将数据按照随机数的顺序重新排列
    Label = label[permutation, :]  # 将标记按照随机数的顺序重新排列
    x_train = data[:numTrain]  # 训练集
    x_val = data[numTrain:]  # 验证集
    y_train = Label[:numTrain]  # 训练集的label
    y_val = Label[numTrain:]  # 验证集的label
    x_train = np.expand_dims(x_train,axis=data.ndim)#将数据集扩充维度，输入网络
    x_val = np.expand_dims(x_val,axis=data.ndim)
    return x_train, y_train, x_val, y_val
def smooth_labels(labels, factor=0.02):#标签平滑
    # smooth the labels
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    # returned the smoothed labels
    return labels
#调用函数读取数据
spectrum_data = load_f(r'spectrum_feature.pkl')#属性为list
labels = load_f(r'label.pkl')
mfcc_data = Normalization(spectrum_data)
#分割训练集测试集
x_train, y_train, x_val, y_val = Train_Val_Split(spectrum_data,labels)
y_train = smooth_labels(y_train)
#*****************************模型训练*****************************
#**********定义模型**********
model = models.Sequential()#构建sequential顺序模型
model.add(layers.Conv2D(input_shape=(257,123,1),filters=32, kernel_size=3,strides=2, padding='same',activation='relu',data_format='channels_last',name='CONV1'))
model.add(layers.Dropout(0.6,name='DP1'))
model.add(layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.001),name='CONV2'))
model.add(layers.Dropout(0.6,name='DP2'))#防止过拟合，随机去掉20%的神经元连接
model.add(layers.Conv2D(filters=128, kernel_size=3, strides=2,activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001),name='CONV3'))
model.add(layers.Dropout(0.6,name='DP3'))#防止过拟合，随机去掉20%的神经元连接
model.add(layers.Conv2D(filters=256, kernel_size=3, strides=2, activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.001),name='CONV4'))
model.add(layers.Dropout(0.6,name='DP4'))#防止过拟合，随机去掉20%的神经元连接
model.add(layers.Conv2D(filters=512, kernel_size=3, strides=2, activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.001),name='CONV5'))
# model.add(layers.Dropout(0.5,name='DP5'))#防止过拟合，随机去掉20%的神经元连接
model.add(layers.Flatten(name='FLT'))#将多维变量变为二维变量，因全连接层的输入只能为二维变量
model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dropout(0.2))   # 0.2是神经元舍弃比
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(6, activation='softmax',name='FC1'))#一共有6类情感，softmax分类器输出
plot_model(model, to_file='spectrum_model.png', show_shapes=True)#做出2D_CNN结构图
model.summary()
#**********编译模型**********
#配置学习过程 （多分类问题）
opt = optimizers.RMSprop(lr=0.0001, decay=1e-6)#定义RMS优化器，用于优化损失函数
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])#定义损失函数ADAMval

callbacks_list = [keras.callbacks.EarlyStopping(monitor='accuracy',patience=50,verbose=0,mode='auto')]#提前停止
keras.callbacks.ModelCheckpoint(filepath='speechmfcc_model_checkpoint.h5',monitor='val_loss',save_best_only=True)#该回调函数将在每个epoch后保存模型到filepath

#history = model.fit(x_train, y_train, epochs=50,batch_size=16,validation_data=(x_val, y_val))#模型训练
history = model.fit(x_train, y_train,epochs=150,batch_size=64,validation_data=(x_val, y_val),shuffle=True,callbacks=callbacks_list)

model.save('spectrum_model.h5')
#model.save_weights('mfcc_model_weight.h5')


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
ax2.set_title('spectrum model acc')
ax2.set_ylabel('acc')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'test'], loc='upper left')
plt.show()

print(history.history['val_accuracy'])
print(history.history['accuracy'])
