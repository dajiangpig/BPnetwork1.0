# coding=gbk
import h5py
import sklearn.datasets
import sklearn.linear_model
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)

font = fm.FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')#这一步是为了引入中文字体
matplotlib.rcParams['savefig.dpi'] = 300#设置图片像素
matplotlib.rcParams['figure.figsize'] = (3,2)

def sigmoid(input_sum):
    """
    函数：
            激活函数Sigmoid
    输入：
            param input_sum: 输入，即神经元的加权和
    返回：
            return:激活后的输出
            input_sum: 把输入缓存起来返回
    """
    output = 1.0/(1+np.exp(-input_sum))
    return output, input_sum

def sigmoid_back_propagation(derror_wrt_output, input_sum):
    """
    函数：
            误差关于神经元输入的偏导: dE/dIn = dE/dOut * dOut/dIn  PDF公式5.6
            其中： dOut/dIn   就是激活函数的导数  dy=y(1-y),PDF公式（5.9）
    输入：
         derror_wrt_output：误差关于神经元输出的偏导：公式（5.7）
         input_sum: 输入加权和
    返回：
        derror_wrt_dinputs: 误差关于输入的偏导，见式（5.13）
    """
    output = 1.0/(1 + np.exp(- input_sum))
    doutput_wrt_dinput = output * (1 - output)
    derror_wrt_dinput = derror_wrt_output * doutput_wrt_dinput

    return derror_wrt_dinput

def relu(input_sum):
    """
    函数：
            激活函数ReLU
    输入：
            input_sum:输入，即神经元的加权和
    返回：
            outputs:激活后的输出
            input_sum：把输入缓存起来返回
    """
    output = np.maximum(0,input_sum)
    return output, input_sum

def relu_back_propagation(derror_wrt_output, input_sum):
    """
    函数：
            误差关于神经网络输入的偏导：dE/dIn = dE/dOut * dOut/dIn
            其中：dOut/dIn 就是激活函数的导数
                  dE/dOut 误差对神经元输出的偏导
    输入：
            derror_wrt_output:误差关于神经元输出的偏导
            input_sum: 输入加权和
    返回：
            derror_wrt_dinputs: 误差关于输入的偏导
     ??? 最好再check 一下地下那个公式能不能用
    """
    derror_wrt_dinputs = np.array(derror_wrt_output,copy=True)
    #derror_wrt_dinputs[input_sum <= 0] = 0
    for i in input_sum:
        if i <= 0:
            derror_wrt_dinputs[i] = 0
    return derror_wrt_dinputs

def tanh(input_sum):
    """
    函数：
            激活函数 tanh
    输入：
            input_sum:输入，即神经元的加权和
    返回：
            output: 激活后的输出
            input_sum：把输入缓存起来返回
    """
    output = np.tanh(input_sum)
    return output, input_sum

def tanh_back_propagation(derror_wrt_output, input_sum):
    """
    函数：
            误差关于神经元输入的偏导：dE/dIn = dE/dOut * dOut/dIn
            其中：dOut/dIn 就是激活函数的导数 tanh'(x) = 1- x^2
                  dE/dOut 误差对神经元输出的偏导
    输入：
            derror_wrt_output:误差关于神经元输出的偏导:公式（5.7）
            input_sum: 输入加权和
    返回：
            derror_wrt_dinputs:误差关于输入的偏导
    """
    output = np.tanh(input_sum)
    doutput_wrt_dinput = 1 - np.power(output,2)
    derror_wrt_dinput = derror_wrt_output * doutput_wrt_dinput

    return derror_wrt_dinput

def activiated(activation_choose,input):
    """把正向激活包装一下"""
    if activation_choose == "sigmoid":
        return sigmoid(input)
    elif activation_choose == "relu":
        return relu(input)
    elif activation_choose == "tanh":
        return tanh(input)
    return sigmoid(input)#相当于默认

def activated_back_propagation(activation_choose, derror_wrt_output,input):
    """
     把反向激活函数包装一下
    """
    if activation_choose == "sigmoid":
        return sigmoid_back_propagation(derror_wrt_output,input)
    elif activation_choose == "relu":
        return relu_back_propagation(derror_wrt_output,input)
    elif activation_choose == "tanh":
        return tanh_back_propagation(derror_wrt_output,input)
    return sigmoid_back_propagation(derror_wrt_output,input)
