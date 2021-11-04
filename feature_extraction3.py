#!/usr/bin/env python
# -*- conding:utf-8 -*-
# Author: hum741
# CreatTime: 2021/10/5
# FileName: feature_extraction3
# Description:基于语谱图和mfcc和lstm等特征合并为一维特征
#!/usr/bin/env python
#-*- conding:utf-8 -*-
# Author: hum741
# CreatTime: 2021/7/27
# FileName: Feature_map_extract
# Description: 基于MFCC深层特征和IMFCC深层特征叠加得到合成特征

import matplotlib.pyplot as plt
import numpy as np
import os #文件处理模块
import pickle #读取、存储文件
from keras.models import load_model
from keras import backend as K
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

def zscoreNormalization(x):
    x=(x-np.mean(x,axis=0))/np.std(x,axis=0)
    return x

with open('mfcc_feature.pkl', 'rb') as f:
    mfcc_data = pickle.load(f)#读取上步保存的特征文件
with open('imfcc_feature.pkl', 'rb') as f:
    imfcc_data= pickle.load(f)#读取上步保存的特征文件
with open('spectrum_feature.pkl','rb') as f:
    spect_data = pickle.load(f)
with open('mel_spectrum_feature.pkl','rb') as f:
    mel_spect_data = pickle.load(f)
with open('label.pkl','rb') as f:
    labels = pickle.load(f)
mfcc_data = np.array(mfcc_data)
imfcc_data = np.array(imfcc_data)
spect_data = np.array(spect_data)
mel_spect_data = np.array(mel_spect_data)
mfcc_data = mfcc_data.reshape((mfcc_data.shape[0], mfcc_data.shape[1], mfcc_data.shape[2]))#对数据尺寸进行规整
imfcc_data = imfcc_data.reshape((imfcc_data.shape[0],imfcc_data.shape[1],imfcc_data.shape[2]))
spect_data = spect_data.reshape((spect_data.shape[0],spect_data.shape[1],spect_data.shape[2]))
mel_spect_data = mel_spect_data.reshape((mel_spect_data.shape[0],mel_spect_data.shape[1],mel_spect_data.shape[2]))
labels = np.array(labels)
# MFCC_MEAN = np.mean(mfcc_data, axis=0)#压缩行，对列求平均值，返回一个1*126*126*4的矩阵
# MFCC_STD = np.std(mfcc_data, axis=0)#计算每一列的标准差,返回一个1*126*126*4的矩阵
# IMFCC_MEAN = np.mean(imfcc_data,axis=0)
# IMFCC_STD = np.std(imfcc_data,axis=0)
# SPECT_MEAN = np.mean(spect_data, axis=0)#压缩行，对列求平均值，返回一个1*126*126*4的矩阵
# SPECT_STD = np.std(spect_data, axis=0)
# #print(np.shape(DATA_MEAN))
# #print(np.shape(DATA_STD))
# mfcc_data -= MFCC_MEAN#减去平均值
# mfcc_data /= MFCC_STD#除以标准差，数据标准化处理
# imfcc_data-= IMFCC_MEAN
# imfcc_data/= IMFCC_STD
# spect_data -= SPECT_MEAN#减去平均值
# spect_data /= SPECT_STD
mfcc_data = zscoreNormalization(mfcc_data)
imfcc_data = zscoreNormalization(imfcc_data)
spect_data = zscoreNormalization(spect_data)
mel_spect_data = zscoreNormalization(mel_spect_data)
feature_map1 = []#不含LSTM
feature_map2 = []#含LSTM
#加载训练好的模型进行预测
mfcc_model = load_model('mfcc_model.h5')
layer_1 = K.function([mfcc_model.layers[0].input], [mfcc_model.layers[7].output])
imfcc_model = load_model('imfcc_model.h5')
layer_2 = K.function([imfcc_model.layers[0].input], [imfcc_model.layers[7].output])
spect_model = load_model('spectrum_model.h5')
layer_3 = K.function([spect_model.layers[0].input], [spect_model.layers[9].output])
mel_spect_model = load_model('mel_spectrum_model.h5')
layer_4 = K.function([mel_spect_model.layers[0].input],[mel_spect_model.layers[9].output])
for i in range(1200):
    mfcc_feature_map = layer_1([np.expand_dims(mfcc_data[i,:,:],axis = 0)])[0]  # 数据维度为126*126*1*16
    mfcc_feature_map = np.squeeze(mfcc_feature_map)
    mfcc_feature_map =mfcc_feature_map.reshape(1536)
    # print(np.shape(mfcc_feature_map))
    imfcc_feature_map = layer_2([np.expand_dims(imfcc_data[i,:,:],axis = 0)])[0]
    imfcc_feature_map = np.squeeze(imfcc_feature_map)
    imfcc_feature_map = imfcc_feature_map.reshape(1536)

    spect_feature_map = layer_3([np.expand_dims(spect_data[i,:,:],axis = 0)])[0]
    spect_feature_map = np.squeeze(spect_feature_map)
    spect_feature_map = spect_feature_map.reshape(18432)

    mel_spect_feature_map = layer_4([np.expand_dims(mel_spect_data[i,:,:],axis=0)])[0]
    mel_spect_feature_map = np.squeeze(mel_spect_feature_map)
    mel_spect_feature_map = mel_spect_feature_map.reshape(6144)
    feature_map_temp1 = np.concatenate((mfcc_feature_map, imfcc_feature_map), axis=0)#不含频谱图
    feature_map_temp2 = np.concatenate((spect_feature_map,mel_spect_feature_map), axis=0)
    feature_map1.append((feature_map_temp1))
    feature_map2.append((feature_map_temp2))

with open('deep_feature_map1.pkl', 'wb') as f:
    pickle.dump(feature_map1, f)#1200*445568
with open('deep_feature_map2.pkl', 'wb') as f:
    pickle.dump(feature_map2, f)#1200*955456
print(np.shape(feature_map1))
print(np.shape(feature_map2))

# #提取深层特征
# for _ in range(64):
#     show_img = mfcc_feature_map[ :, :, _]
#     show_img.shape = [6, 6]
#     plt.subplot(8, 8, _ + 1)
#     plt.subplot(8, 8, _ + 1)
#     plt.imshow(show_img, cmap='gray')
#     plt.axis('off')
# plt.show()
# for _ in range(64):
#     show_img = imfcc_feature_map[ :, :, _]
#     show_img.shape = [6, 6]
#     plt.subplot(8, 8, _ + 1)
#     plt.subplot(8, 8, _ + 1)
#     plt.imshow(show_img, cmap='gray')
#     plt.axis('off')
# plt.show()
print(np.shape(mfcc_feature_map))