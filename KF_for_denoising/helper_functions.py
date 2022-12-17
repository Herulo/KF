# -*- coding: utf-8 -*-
"""
卡尔曼滤波用于语音降噪 相关函数
2022-12-26
作者：王猛、王沛元 云南大学信息学院
"""

import numpy as np
from scipy.linalg import toeplitz


def awgn(signal, SNR_db):
    # 为输入信号添加加性高斯白噪声
    S = np.sum(signal ** 2)  # 信号功率
    wg_noise = np.random.randn(len(signal), 1)  # 塑造等长噪声序列
    N = np.sum(wg_noise ** 2)  # 噪声功率
    wg_noise = np.sqrt(10 ** (-SNR_db / 10) * (S / N)) * wg_noise  # 高斯白噪声
    noisy_signal = signal + wg_noise  # 加噪音频
    return noisy_signal, wg_noise


def sliding_window(signal, Fs):
    # 对音频分帧加窗处理（基于平稳的假设）

    # 50%覆盖参数设置
    stationary_samples = int(30e-3 * Fs)  # 帧长
    # print(stationary_samples)
    half_stationary = int(stationary_samples / 2)  # 半帧长

    # 补0
    padding = half_stationary - np.remainder(len(signal), half_stationary)
    signal_padding = np.concatenate((signal, np.zeros((padding, 1))))

    # 分帧
    signal_sliced_half_stationary = np.reshape(np.concatenate((signal_padding, np.zeros((half_stationary, 1)))),
                                               (half_stationary, -1), order='F')
    # print(signal_sliced_half_stationary.shape)
    signal_sliced_half_stationary_delayed = np.reshape(np.concatenate((np.zeros((half_stationary, 1)), signal_padding)),
                                                       (half_stationary, -1), order='F')
    # print(signal_sliced_half_stationary_delayed.shape)
    signal_sliced = np.concatenate((signal_sliced_half_stationary_delayed, signal_sliced_half_stationary), axis=0)
    # print(signal_sliced.shape)
    signal_sliced = signal_sliced[:, 1:-1]  # 把前后补0帧去掉
    # print(signal_sliced.shape)

    # 加窗
    ham_window = np.hamming(stationary_samples)
    ham_window = ham_window.reshape((len(ham_window), 1))
    signal_sliced_windowed = ham_window @ np.ones((1, signal_sliced.shape[1])) * signal_sliced

    return signal_sliced_windowed, padding


def sliding_window_rec(signal_sliced_windowed, Fs, padding):
    # 对音频分帧加窗处理（基于平稳的假设）

    # 50%覆盖参数设置
    stationary_samples = int(30e-3 * Fs)  # 帧长
    half_stationary = int(stationary_samples / 2) # 半帧长

    x = signal_sliced_windowed.shape[1]
    signal_reconstructed = np.zeros((half_stationary * (x + 1), x))

    # 展开
    ham_window = np.hamming(stationary_samples)
    ham_window = ham_window.reshape((len(ham_window), 1))
    unwindow = 1 / (ham_window[0:half_stationary] + ham_window[half_stationary:])
    unwindow = unwindow.reshape((len(unwindow), 1))
    unwindow_2 = np.repeat(unwindow, 2)
    unwindow_2 = unwindow_2.reshape((len(unwindow_2), 1))

    for ind in range(x):
        a = np.zeros((half_stationary * ind, 1))
        b = signal_sliced_windowed[:, ind:ind + 1] * unwindow_2
        c = np.zeros((half_stationary * (x - ind - 1), 1))
        signal_reconstructed[:, ind:ind + 1] = np.concatenate((a, b, c), axis=0)

    signal_reconstructed2 = np.sum(signal_reconstructed, axis=1)
    signal_reconstructed2 = signal_reconstructed2.reshape((len(signal_reconstructed2), 1))
    signal_reconstructed = signal_reconstructed2[0:-padding]

    return signal_reconstructed


def Yule_Walker(x, order):
    # 计算转移矩阵系数

    rxx = np.correlate(x[0, :], x[0, :], "full")  # 计算自相关
    rxx = rxx.reshape((len(rxx), 1))
    zero = x.shape[1]  # 保留长度
    # 初始化矩阵
    rxx_vector = rxx[zero + 1:zero + order + 1]
    Rxx = toeplitz(rxx[zero:zero + order])
    a = np.concatenate((np.ones(((1, 1))), np.linalg.inv(Rxx) @ rxx_vector), axis=0)
    row = np.arange(0, order + 1) + zero
    Rxx_row = rxx[row]

    # 计算方差
    var_bruit = 0
    for pp in range(len(a)):
        var_bruit += a[pp] * Rxx_row[pp]

    return a, int(var_bruit)


def kalman_ite(x, P, y, Q, R, Phi, H):
    # 预测
    x = Phi @ x  # 状态向量预估
    P = Phi @ P @ Phi.T + Q  # 预测误差矩阵

    # 卡尔曼增益计算
    K = (P @ H.T) @ np.linalg.inv(H @ P @ H.T + R)  # Kalman gain

    # 更新
    x = x + K @ (y - H @ x)  # 状态向量校正估计
    P = P - K @ H @ P  # 后验误差矩阵
    return x, P
