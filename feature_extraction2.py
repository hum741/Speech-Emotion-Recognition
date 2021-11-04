#!/usr/bin/env python
# -*- conding:utf-8 -*-
# Author: hum741
# CreatTime: 2021/10/4
# FileName: feature_extraction2
# Description:在特征提取模块中添加频谱图提取
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import wave
import os #文件处理模块
import pickle #读取、存储文件
from numpy import expand_dims


#定义函数
def normalizeVoiceLen(y, normalizedLen):
    nframes = len(y)#得到音频长度
    y = np.reshape(y, [nframes, 1]).T#将音频信号表示为一维数组，列向量转秩为行向量
    # 归一化音频长度为2s,32000数据点
    if (nframes < normalizedLen):
        res = normalizedLen - nframes
        res_data = np.zeros([1, res], dtype=np.float32)#如果音频长度不够，补零
        y = np.reshape(y, [nframes, 1]).T#列向量转秩为行向量
        y = np.c_[y, res_data]#拼接两个数组。
    else:
        y = y[:, 0:normalizedLen]#将音频长度归一化为normalizedLen的长度
    return y[0]

def getNearestLen(framelength, sr):
    framesize = framelength * sr#帧长度和采样率相乘
    # 找到与当前framesize最接近的2的正整数次方
    nfftdict = {}
    lists = [32, 64, 128, 256, 512, 1024]
    for i in lists:
        nfftdict[i] = abs(framesize - i)
    sortlist = sorted(nfftdict.items(), key=lambda x: x[1])  # 按与当前framesize差值升序排列
    framesize = int(sortlist[0][0])  # 取最接近当前framesize的那个2的正整数次方值为新的framesize
    return framesize
def PreEmphasised(x):#语音预加重
    PointNumbers=len(x)
    PreEmphasis=x
    PointNumbers=int(PointNumbers)#转化为整型
    for i in range(1,PointNumbers,1):#range(PointNumbers)，PointNumbers需为整型
        PreEmphasis[i]=PreEmphasis[i]-0.97*PreEmphasis[i-1]
    PreEmphasis =PreEmphasis.astype(np.float)
    return (PreEmphasis)
def zscoreNormalization(x):
    x=(x-np.mean(x))/np.std(x)
    return x
def normalize_pict(y):#对特征图进行0-255归一化
    array = (y - np.min(y)) / (np.max(y) - np.min(y))
    return array
def mfcc_extract(y):#这个提取mfcc的函数不如下面那个准确
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=39, n_fft=N_FFT, hop_length=int(N_FFT / 2))
    #mfcc_feature = normalize_pict(mfcc_feature)
    #mfcc_feature = expand_dims(mfcc_feature, axis=0)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_delta = librosa.feature.delta(mfcc)
    feature = np.concatenate((mfcc, mfcc_delta, mfcc_delta_delta), axis=0)
    return mfcc
def mfcc_extract_lstm(y):#这个提取mfcc的函数不如下面那个准确
    mfcc_feature = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=126, n_fft=N_FFT, hop_length=int(N_FFT / 2))
    mfcc_feature_lstm = np.mean((mfcc_feature),axis=0)
    #mfcc_feature = normalize_pict(mfcc_feature)
    #mfcc_feature = expand_dims(mfcc_feature, axis=0)
    return mfcc_feature_lstm
def mfcc_extract2(y):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=200, n_fft=N_FFT, hop_length=int(N_FFT / 2))
    mfcc = librosa.feature.mfcc(y, sr, S=librosa.power_to_db(S), n_mfcc=126)  # 提取mfcc系数
    #mfcc = normalize_pict(mfcc)
    mfcc=expand_dims(mfcc,axis=0)
    return mfcc#得到126*126的向量
def imfcc_extract(y):
    S = np.abs(librosa.core.stft(y, n_fft=N_FFT, hop_length=int(N_FFT / 2)))**2.0
    mel_basis = librosa.filters.mel(sr, N_FFT)
    mel_basis = np.linalg.pinv(mel_basis).T
    mel = np.dot(mel_basis, S)
    S = librosa.power_to_db(mel)  # 128*126的矩阵
    dct_temp = librosa.filters.dct(39, S.shape[0])  # 126*128的矩阵
    imfcc = np.dot(dct_temp, S)  # 126*126的矩阵
    imfcc_delta = librosa.feature.delta(imfcc)
    imfcc_delta_delta = librosa.feature.delta(imfcc_delta)
    #接下来进行各矩阵的0-255归一化
    #imfcc = normalize_pict(imfcc)
    #imfcc_delta = normalize_pict(imfcc_delta)
    #imfcc_delta_delta = normalize_pict(imfcc_delta_delta)
    #接下来进行维数扩张
    # imfcc = np.expand_dims((imfcc), axis=0)
    # imfcc_delta = np.expand_dims((imfcc_delta), axis=0)
    # imfcc_delta_delta = np.expand_dims((imfcc_delta_delta), axis=0)
    # #接下来进行通道方向的拼接
    # imfcc_feature = np.concatenate((imfcc, imfcc_delta, imfcc_delta_delta), axis=0)
    return imfcc
def imfcc_extract2(y):

    S = np.abs(librosa.core.stft(y, n_fft=N_FFT ,hop_length=overlapSize)) ** 2.0
    mel_basis = librosa.filters.mel(sr, N_FFT)
    mel_basis = np.linalg.pinv(mel_basis).T
    mel = np.dot(mel_basis, S)
    S = librosa.power_to_db(mel)
    imfcc = np.dot(librosa.filters.dct(13, S.shape[0]), S)
    imfcc_delta = librosa.feature.delta(imfcc)
    imfcc_delta_delta = librosa.feature.delta(imfcc)
    feature = np.concatenate((imfcc, imfcc_delta, imfcc_delta_delta), axis=0)
    return feature


def spectrum_extract1(y):#使用函数提取语谱图
    spectrum, freqs, ts, fig = plt.specgram(y, NFFT=N_FFT, Fs=sr, window=np.hanning(M=N_FFT),
                                            noverlap=overlapSize, scale='dB')  # 绘制频谱图(257*124)
    return spectrum
def spectrum_extract2(y):#按步骤计算提取语谱图
    num_frames = int(np.ceil(float(np.abs(VOICE_LEN - N_FFT)) / overlapSize))  # 向上取整，帧数123
    indicies = np.tile(np.arange(0, N_FFT), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * overlapSize, overlapSize), (N_FFT, 1)).T
    frames = y[indicies.astype(np.int32, copy=False)]  # 123*512的数组，一共123帧，每帧长度为512
    # *************加窗*****************
    frames *= np.hanning(N_FFT)
    # **************进行傅里叶变换并得到功率谱********************
    mag_frames = np.absolute(np.fft.rfft(frames, N_FFT))  # magnitude of FFT
    pow_frames = ((1.0 / N_FFT) * ((mag_frames) ** 2))  # 功率谱(123*257)
    spect = librosa.power_to_db(pow_frames.T, ref=np.max)  # 转化为log形式(257*123)
    return spect,pow_frames

def mel_spect_extract1(y):#使用封装好的函数提取梅尔谱图(128,126)
    mel_spect = librosa.feature.melspectrogram(y, sr=sr, n_fft=N_FFT, hop_length=overlapSize)  # 提取mel特征
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)  # 转化为log形式
    return mel_spect
def mel_spect_extract2(y):
    nfilt = 70
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sr / 2) / 700))  # 将HZ转化为MEL
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))
    bin = np.floor((N_FFT + 1) * hz_points / sr)
    fbank = np.zeros((nfilt, int(np.floor(N_FFT / 2 + 1))))
    spect,pow_frames = spectrum_extract2(y)
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # numerical stability
    filter_banks = 20 * np.log10(filter_banks).T  # (70,123)
    return filter_banks
def LABEL(ed):
    if ed == 'angry':
        Label = 0
    elif ed =='fear':
        Label = 1
    elif ed =='happy':
        Label = 2
    elif ed=='neutral':
        Label = 3
    elif ed =='sad':
        Label =4
    elif ed =='surprise':
        Label = 5
    return Label


emotionDict = {}#设置字典，对字典进行初始化
emotionDict['angry'] = []
emotionDict['fear'] = []
emotionDict['happy'] = []
emotionDict['neutral'] = []
emotionDict['sad'] = []
emotionDict['surprise'] = []

mfccs = {}#创建字典
mfccs['angry'] = []#添加字典条目
mfccs['fear'] = []
mfccs['happy'] = []
mfccs['neutral'] = []
mfccs['sad'] = []
mfccs['surprise'] = []

imfccs = {}
imfccs['angry'] = []#添加字典条目
imfccs['fear'] = []
imfccs['happy'] = []
imfccs['neutral'] = []
imfccs['sad'] = []
imfccs['surprise'] = []

mfccs_lstm = {}
mfccs_lstm['angry'] = []
mfccs_lstm['fear'] = []
mfccs_lstm['happy'] = []
mfccs_lstm['neutral'] = []
mfccs_lstm['sad'] = []
mfccs_lstm['surprise'] = []

Spectrum = {}
Spectrum['angry'] = []
Spectrum['fear'] = []
Spectrum['happy'] = []
Spectrum['neutral'] = []
Spectrum['sad'] = []
Spectrum['surprise'] = []

Mel_Spectrum = {}
Mel_Spectrum['angry'] = []
Mel_Spectrum['fear'] = []
Mel_Spectrum['happy'] = []
Mel_Spectrum['neutral'] = []
Mel_Spectrum['sad'] = []
Mel_Spectrum['surprise'] = []

#MFCC和IMFCC拼接与叠加处理
MFCC_IMFCC_0={}#横向拼接
MFCC_IMFCC_0['angry'] = []
MFCC_IMFCC_0['fear'] = []
MFCC_IMFCC_0['happy'] = []
MFCC_IMFCC_0['neutral'] = []
MFCC_IMFCC_0['sad'] = []
MFCC_IMFCC_0['surprise'] = []
MFCC_IMFCC_1={}#通道方向叠加
MFCC_IMFCC_1['angry'] = []
MFCC_IMFCC_1['fear'] = []
MFCC_IMFCC_1['happy'] = []
MFCC_IMFCC_1['neutral'] = []
MFCC_IMFCC_1['sad'] = []
MFCC_IMFCC_1['surprise'] = []



fileDirCASIA = 'D://Program Files//PyCharm 2020.3.5//Pycharm_Projects//CASIA database'
counter = 0
listdir = os.listdir(fileDirCASIA)#列出数据集的四个文件夹
for persondir in listdir:#一级循环，四个人名命名的文件夹
    if (not r'.' in persondir):
        emotionDirName = os.path.join(fileDirCASIA, persondir)#把每种人名的文件夹路径添加到总数据库路径上
        emotiondir = os.listdir(emotionDirName)#列出人名文件夹下面的情感分类文件夹
        for ed in emotiondir:#二级循环，六种情感文件夹
            if (not r'.' in ed):
                filesDirName = os.path.join(emotionDirName, ed)#继续把情感文件夹的路径添加进去
                files = os.listdir(filesDirName)#列出情感文件夹下面的全部文件
                for fileName in files:
                    if (fileName[-3:] == 'wav'):#找出wav语音文件
                        counter += 1
                        fn = os.path.join(filesDirName, fileName)#把语音文件的名称添加到路径中
                        #print(str(counter) + fn)#编号+路径
                        f = wave.open(fn, 'rb')
                        nchannels, sampwidth, sr, nframes = f.getparams()[:4]  # framerate = 16000,通道数1，帧数42812，采样宽度2
                        # y, sr = librosa.load(fn, sr=None)#读取音频信号值
                        y = f.readframes(nframes)  # 读取音频信号值
                        y = np.frombuffer(y, dtype=np.short)
                        VOICE_LEN = 32000
                        # 获得N_FFT的长度
                        N_FFT = getNearestLen(0.025, sr)
                        overlapSize = int(round(N_FFT / 2))  # 重叠部分采样点数约为每帧长度的1/2，并取整
                        # 统一声音范围为前两秒
                        y = normalizeVoiceLen(y, VOICE_LEN)  # 归一化音频长度为2s,32000个数据点
                        y = y.copy()
                        y = PreEmphasised(y)
                        y = zscoreNormalization(y)
                        mfcc_data = mfcc_extract(y)
                        imfcc_data = imfcc_extract(y)
                        mfcc_lstm = mfcc_extract_lstm(y)
                        spectrum,pow_frames=spectrum_extract2(y)
                        mel_spect = mel_spect_extract2(y)
                        labels = LABEL(ed)
                        labels = np.eye(6)[labels]

                        mfcc_imfcc_0 = np.concatenate((mfcc_data,imfcc_data),axis=0)
                        mfcc_data1 = np.expand_dims((mfcc_data),axis=2)
                        imfcc_data1 = np.expand_dims((imfcc_data),axis=2)
                        mfcc_imfcc_1 = np.concatenate((mfcc_data1,imfcc_data1),axis=2)

                        mfcc_data.tolist()
                        imfcc_data.tolist()
                        mfcc_lstm.tolist()
                        spectrum.tolist()
                        mel_spect.tolist()
                        mfcc_imfcc_0.tolist()
                        mfcc_imfcc_1.tolist()
                        emotionDict[ed].append(labels)
                        mfccs[ed].append(mfcc_data)  # 对应字典中的每种感情添加梅尔特征，tolist把数组变成列表
                        imfccs[ed].append(imfcc_data)
                        mfccs_lstm[ed].append(mfcc_lstm)#直接在这一步保存的话会得到字典格式的特征矩阵
                        Spectrum[ed].append(spectrum)
                        Mel_Spectrum[ed].append(mel_spect)
                        MFCC_IMFCC_0[ed].append(mfcc_imfcc_0)
                        MFCC_IMFCC_1[ed].append(mfcc_imfcc_1)
MFCCS = []
MFCCS = MFCCS + mfccs['angry']
MFCCS = MFCCS + mfccs['fear']
MFCCS = MFCCS + mfccs['happy']
MFCCS = MFCCS + mfccs['neutral']
MFCCS = MFCCS + mfccs['sad']
MFCCS = MFCCS + mfccs['surprise']
MFCCS_LSTM = []
MFCCS_LSTM = MFCCS_LSTM + mfccs_lstm['angry']
MFCCS_LSTM = MFCCS_LSTM + mfccs_lstm['fear']
MFCCS_LSTM = MFCCS_LSTM + mfccs_lstm['happy']
MFCCS_LSTM = MFCCS_LSTM + mfccs_lstm['neutral']
MFCCS_LSTM = MFCCS_LSTM + mfccs_lstm['sad']
MFCCS_LSTM = MFCCS_LSTM + mfccs_lstm['surprise']
IMFCCS = []
IMFCCS = IMFCCS + imfccs['angry']
IMFCCS = IMFCCS + imfccs['fear']
IMFCCS = IMFCCS + imfccs['happy']
IMFCCS = IMFCCS + imfccs['neutral']
IMFCCS = IMFCCS + imfccs['sad']
IMFCCS = IMFCCS + imfccs['surprise']
SPECTRUM = []
SPECTRUM = SPECTRUM + Spectrum['angry']
SPECTRUM = SPECTRUM + Spectrum['fear']
SPECTRUM = SPECTRUM + Spectrum['happy']
SPECTRUM = SPECTRUM + Spectrum['neutral']
SPECTRUM = SPECTRUM + Spectrum['sad']
SPECTRUM = SPECTRUM + Spectrum['surprise']
MEL_SPECTRUM = []
MEL_SPECTRUM = MEL_SPECTRUM + Mel_Spectrum['angry']
MEL_SPECTRUM = MEL_SPECTRUM + Mel_Spectrum['fear']
MEL_SPECTRUM = MEL_SPECTRUM + Mel_Spectrum['happy']
MEL_SPECTRUM = MEL_SPECTRUM + Mel_Spectrum['neutral']
MEL_SPECTRUM = MEL_SPECTRUM + Mel_Spectrum['sad']
MEL_SPECTRUM = MEL_SPECTRUM + Mel_Spectrum['surprise']

MFCC_IMFCC0 = []
MFCC_IMFCC0 = MFCC_IMFCC0 + MFCC_IMFCC_0['angry']
MFCC_IMFCC0 = MFCC_IMFCC0 + MFCC_IMFCC_0['fear']
MFCC_IMFCC0 = MFCC_IMFCC0 + MFCC_IMFCC_0['happy']
MFCC_IMFCC0 = MFCC_IMFCC0 + MFCC_IMFCC_0['neutral']
MFCC_IMFCC0 = MFCC_IMFCC0 + MFCC_IMFCC_0['sad']
MFCC_IMFCC0 = MFCC_IMFCC0 + MFCC_IMFCC_0['surprise']

MFCC_IMFCC1 = []
MFCC_IMFCC1 = MFCC_IMFCC1 + MFCC_IMFCC_1['angry']
MFCC_IMFCC1 = MFCC_IMFCC1 + MFCC_IMFCC_1['fear']
MFCC_IMFCC1 = MFCC_IMFCC1 + MFCC_IMFCC_1['happy']
MFCC_IMFCC1 = MFCC_IMFCC1 + MFCC_IMFCC_1['neutral']
MFCC_IMFCC1 = MFCC_IMFCC1 + MFCC_IMFCC_1['sad']
MFCC_IMFCC1 = MFCC_IMFCC1 + MFCC_IMFCC_1['surprise']



Labels = []  # 读取Label数据
Labels = Labels + emotionDict['angry']
Labels = Labels + emotionDict['fear']
Labels = Labels + emotionDict['happy']
Labels = Labels + emotionDict['neutral']
Labels = Labels + emotionDict['sad']
Labels = Labels + emotionDict['surprise']
with open('mfcc_feature.pkl', 'wb') as f: #打开文件，写入二进制文本
    pickle.dump(MFCCS, f) #封装，将mfccs写入f中
with open('mfcc_lstm_feature.pkl','wb') as f:
    pickle.dump(MFCCS_LSTM,f)
with open('imfcc_feature.pkl', 'wb') as f: #打开文件，写入二进制文本
    pickle.dump(IMFCCS, f) #封装，将mfccs写入f中
with open('mfcc_imfcc_0.pkl','wb') as f:
    pickle.dump(MFCC_IMFCC0,f)
with open('mfcc_imfcc_1.pkl','wb') as f:
    pickle.dump(MFCC_IMFCC1,f)
with open('spectrum_feature.pkl','wb') as f:
    pickle.dump(SPECTRUM,f)
with open('mel_spectrum_feature.pkl','wb') as f:
    pickle.dump(MEL_SPECTRUM,f)
with open('label.pkl', 'wb') as f: #打开文件，写入二进制文本，label1未转化为热独编码，pytorch使用
    pickle.dump(Labels, f) #封装，将mfccs写入f中
    f.close()
print(np.shape(MFCCS))#(1200,126,126)
print(np.shape(IMFCCS))#(1200,126,126)
print(np.shape(MFCCS_LSTM))#(1200,126)
print(np.shape(Labels))
print(np.shape(SPECTRUM))#(1200,257,123)
print(np.shape(MEL_SPECTRUM))#(1200,70,123)
print(np.shape(MFCC_IMFCC0))
print(np.shape(MFCC_IMFCC1))

#将MFCC和IMFCC进行拼接与叠加处理


