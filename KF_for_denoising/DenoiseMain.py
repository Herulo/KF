# -*- coding: utf-8 -*-
"""
卡尔曼滤波用于语音降噪
2022-12-26
作者：王猛、王沛元 云南大学信息学院
"""

# 关联包
# import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import librosa
# 自建函数
from helper_functions import awgn, sliding_window, sliding_window_rec, Yule_Walker, kalman_ite

plt.rc("font", family='Microsoft YaHei')  # 设置字体

# 加载信号
signal, sample_rate = librosa.load("oldman.wav", sr=8000, duration=10)
signal = signal.reshape(-1, 1)
# print("输入信号类型：",signal.shape)

# 参数设置
Fs = 8000  # 采样率 默认8k
N = signal.shape[0]  # 信号长度获取

# 时间轴设置
t = np.arange(0, N / Fs, 1 / Fs)
# print(len(t))
t = np.reshape(t, (len(t), 1))
# print(t.shape)

# 自回归阶数及迭代次数
p = 16  # AR order default=16
ite_kalman = 1  # 设置卡尔曼滤波迭代次数

# 添加噪声
SNR = 10  # 信噪比
noisy_signal, wg_noise = awgn(signal, SNR)  # 加性高斯白噪声

# 绘制原始音频与加噪音频
plt.figure()
plt.grid()
plt.title('原始信号与加噪后信号对比')
plt.plot(t, noisy_signal, 'r--', linewidth=0.7)
plt.plot(t, signal)
plt.xlabel('时间(s)')
plt.ylabel('幅度')
plt.legend(['加噪信号', '原始信号'])
plt.show()

# 噪声方差评估 (基于静默段)
varNoise = np.var(noisy_signal[-1000:-1])

# 信号分帧加窗
signal_sliced_windowed, padding = sliding_window(signal, Fs)

# 分帧加窗后维度保存中间变量
signal_sliced_windowed_filtered = np.zeros((signal_sliced_windowed.shape))

for ite_slice in range(signal_sliced_windowed.shape[1]):

    # 对于每个分段处理
    slice_signal = signal_sliced_windowed[:, ite_slice:ite_slice + 1].T

    # 卡尔曼滤波迭代
    for ite in range(ite_kalman):
        a, var_bruit = Yule_Walker(slice_signal, p)  # 转移矩阵参数估计
        signal_filtered = np.zeros((1, signal_sliced_windowed.shape[0]))  # 信号保存

        # 构造 Phi 和 H
        Phi = np.concatenate((np.zeros((p - 1, 1)), np.eye(p - 1)), axis=1)
        Phi = np.concatenate((Phi, -np.fliplr(a[1:].T)), axis=0)
        H = np.concatenate((np.zeros((p - 1, 1)), np.ones((1, 1))), axis=0).T

        # 构造 Q , R , Po
        Q = var_bruit * np.eye(p)
        R = varNoise
        P = 10000 * np.eye(p)

        # 状态向量初始化
        x = np.zeros((p, 1))

        for js in range(signal_sliced_windowed.shape[0]):
            y = slice_signal[0][js]  # 观察值
            [x, P] = kalman_ite(x, P, y, Q, R, Phi, H) #卡尔曼滤波
            signal_filtered[0][js] = x[-1]

        slice_signal = signal_filtered #滤波后信号
    signal_sliced_windowed_filtered[:, ite_slice:ite_slice + 1] = signal_filtered.T

# 重构成信号
signal_reconstructed = sliding_window_rec(signal_sliced_windowed_filtered, Fs, padding)

# 绘制加噪信号与重构信号对比图
plt.figure()
plt.grid()
plt.title('加噪信号与重构信号对比')
plt.plot(t, noisy_signal)
plt.plot(t, signal_reconstructed, 'r--', linewidth=0.5)
plt.xlabel('时间(s)')
plt.ylabel('幅度')
plt.legend(['加噪信号', '重构信号'])
plt.show()

# 音频效果播放
# sd.play(signal, Fs, blocking=True)
# sd.play(wg_noise, Fs, blocking=True)
sd.play(noisy_signal, Fs, blocking=True)
sd.play(signal_reconstructed, Fs, blocking=True)
