#这里估计是写主函数
from BPActivationPart import *

class NeuralNetwork:
    def __init__(self, layers_strcuture, print_cost = False):
        self.layers_strcuture = layers_strcuture
        self.layers_num = len(layers_strcuture)

        #除掉输入层的网络层数，因为其他层才是真正的神经元层
        self.param_layers_num = self.layers_num -  1

        self.learning_rate = 0.0618#学习率
        self.num_iterations = 2000#迭代次数
        self.x = None
        self.y = None
        self.w = dict()#权重
        self.b = dict()#偏差
        self.costs = []
        self.print_cost = print_cost

        self.init_w_and_b()
    def set_learning_rate(self, learning_rate):
        """"设置学习率"""
        self.learning_rate = learning_rate

    def set_num_iterations(self,num_iterations):
        """设置迭代次数"""
        self.num_iterations = num_iterations

    def set_xy(self, input, expected_output):
        """"设置神经网络期望的输入和期望的输出"""
        self.x = input
        self.y = expected_output

    def init_w_and_b(self):
        """
        函数
            初始化神经网络所有参数
        输入：
            layers_strcuture ：神经网络的结构。例如[2，4，3，1]，4层结构：
            第0层输入层接收2个数据，第1层隐藏层4个神经元，第2层隐藏层3个神经元，第3层输出层1个神经元
        返回：
            神经网络各层参数的索引表，用来定位权值 wi 和偏置 bi, i 为网络层编号
        """
        np.random.seed(3)
        """
        当前神经元层的权值为 n_i * n_i-1的矩阵，i 为网络层编号，n为下标i代表的网络层的节点个数
        例如[2，4，3，1]，4层结构：第0层输入层为2，那么第1层隐藏层神经元的个数为4
        那么第1层权值w是一个4*2的矩阵，如：
            w1 = array([ [-0.9,-0.5],
                         [0.5,0.45],
                         [-0.022,0.13],
                         [-0.079,-1.498] ])
        当前层的偏置一般给0就行，偏置是个1*ni的矩阵，ni为第i层的节点个数，例如第1层为4个节点，那么：
        b1 = array([0., 0., 0., 0.])
        """
        for l in range(1,self.layers_num):
            self.w["w" + str(l)] = np.random.randn(self.layers_strcuture[l],
                                                   self.layers_strcuture[l-1])/np.sqrt(self.layers_strcuture[l-1])
            self.b["b" + str(l)] = np.zeros((self.layers_strcuture[l],1))

        return self.w, self.b
    def layer_activation_forward(self, x, w, b,activation_choose):
        """
        函数：
            网络的正向传播
        输入：
            x：当前网络层输入（即上一层的输出），一般是所有训练数据，即输入矩阵
            w: 当前网络层的权值矩阵
            b: 当前网络层的偏置矩阵
            activation_choose: 选择激活函数"sigmoid","relu","tanh"
        返回：
            output:网络层的激活输出
            cache：缓存该网络层的信息，供后续使用：（x,w,b,input_sum）-> cache
        """
        # dot为正常矩阵求法，并不是点乘，此步骤为对输入求加权和
        input_sum = np.dot(w, x) + b
        # 对输入加权和进行激活输出
        output,_ = activiated(activation_choose, input_sum)

        return output, (x,w,b,input_sum)

    def forward_propagation(self,x):
        """
        函数：
            神经网络的正向传播
        输入：
            待处理的数据
        返回：
            output: 正向传播完成后的输出层的输出
            caches：正向传播过程中，缓存每一个网络层的信息，（x,w,b,input_sum）-> cache
        """
        caches = []
        #作为输入层，输出=输入
        output_prev = x
        # 第0层作为输入层，只负责观察到输入的数据，并不需要处理，正向传播从第1层开始，一直到输出层输出为止
        # range(1, n) => [1, 2, ..., n-1]
        L = self.param_layers_num #真实的一共有多少层
        for l in range(0,L - 1):
            # 当前网络的输入来自上一层的输出
            input_cur = output_prev
            output_prev, cache = self.layer_activation_forward(input_cur,self.w["w"+str(l+1)],self.b["b"+str(l+1)],"tanh")
            caches.append(cache)
        output,cache = self.layer_activation_forward(output_prev,self.w["w" + str(L)],self.b["b" + str(L)],"sigmoid")
        caches.append(cache)

        return output, caches

    def show_caches(self,caches):
        """显示网络层的缓存参数信息"""
        i = 1
        for cache in caches:
            print("%dtd Layer" % i)
            print("input:%"% cache[0])
            print("w: %s" % cache[1])
            print("b: %s" % cache[2])
            print(" input_sum:%s" % cache[3])
            print("---------华丽的分割线")
            i += 1
    def compute_error(self,output):
        """
        函数:
            计算当次迭代的输出总误差
        输入：
        返回:
        """
        m = self.y.shape[1]
        # 计算误差，见式5.5
        # error = -np.sum(0.5 * (self.y - output)^2)/m

        # 交叉熵作为误差函数
        error = -np.sum(np.log(output) * self.y + np.log(1-output) * (1-self.y))/ m
        error = np.squeeze(error)
        return error

    def layer_activation_backward(self,derror_wrt_output,cache,activation_choose):
        """
           函数:
                网络层的反向传播
           输入：
                derror_wrt_output:误差关于输出的偏导
                cache: 网络层的缓存信息（x,w,b,input_sum）
                activation_choose:选择激活函数“sigmoid”,"relu","tanh"
           返回：梯度信息，：即
                derror_wrt_output_prev:反向传播到上一层的误差关于输出的梯度
                derror_wrt_dw: 误差关于权值的梯度
                derror_wrt_db:误差关于偏置的梯度
        """
        input,w,b,input_sum = cache
        output_prev = input # 上一层的输出 = 当前层的输入； 注意是'输入'不是输入的加权和（input_sum）
        m = output_prev.shape[1] #m 是输入样本的数量，我们要取均值，所以下面求值要除以m

        # 实现式（5.13）->误差关于权值w的偏导数
        derror_wrt_dinput = activated_back_propagation(activation_choose,derror_wrt_output,input_sum)
        derror_wrt_dw = np.dot(derror_wrt_dinput,output_prev.T)/m ###乘以一个转置就可以了吗？回去再关注一下

        # 实现式（5.32）->误差关于偏f置b的偏导数
        derror_wrt_db = np.sum(derror_wrt_dinput,axis = 1, keepdims = True)/m

        # 为反向传播到上一层提供误差传递，见式（5.28）部分
        derror_wrt_output_prev = np.dot(w.T,derror_wrt_dinput)

        return derror_wrt_output_prev, derror_wrt_dw, derror_wrt_db

    def back_propagation(self,output,caches):
        """
        函数：
             神经网络反向传播
        输入：
             output:神经网络输出
             caches : 所有网络层（输入层不算）的缓存参数信息[（x,w,b,input_sum）]
        返回:

        """
        grads = {}
        L = self.param_layers_num
        # 把输出层输出重构成和期望输出一样的结构
        output = output.reshape(output.shape)

        expected_output = self.y
        # 见式（5.8）
        # derror_wrt_output = -(expected_output - output)
        # 交叉熵作为误差函数
        derror_wrt_output = -(np.divide(expected_output,output) - np.divide(1-expected_output,1-output))

        # 反向传播:输出层-> 隐藏层，得到梯度：见式（5.8），（5.13），（5.15）
        #取最后一层，即输出层的参数信息
        current_cache = caches[L-1]
        grads["derror_wrt_output" + str(L)],grads["derror_wrt_dw" + str(L)],grads["derror_wrt_db" + str(L)] = \
            self.layer_activation_backward(derror_wrt_output,current_cache,"sigmoid")
        # 反向传播：隐藏层—〉隐藏层，得到梯度：见式（5.28），（5.32）
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            derror_wrt_output_prev_temp,derror_wrt_dw_temp,derror_wrt_db_temp = \
                self.layer_activation_backward(grads["derror_wrt_output" + str(l+2)],current_cache,"tanh")
            grads["derror_wrt_output" + str(l+1)] = derror_wrt_output_prev_temp
            grads["derror_wrt_dw" + str(l+1)] = derror_wrt_dw_temp
            grads["derror_wrt_db" + str(l+1)] = derror_wrt_db_temp

        return grads

    def update_w_and_b(self, grads):
        """
        函数：
            根据梯度'信息更新w,b
        输入：
            grads:当前迭代的梯度信息
        返回：
        """
        #权值w和偏置b的更新，见公式：（5.16），（5.18）
        for l in range(self.param_layers_num):
            self.w["w" + str(l+1)] = self.w["w" + str(l+1)] - self.learning_rate * grads["derror_wrt_dw" + str(l+1)]
            self.b["b" + str(l + 1)] = self.b["b" + str(l+1)] - self.learning_rate * grads["derror_wrt_db" + str(l+1)]

    def traning_model(self):
        """训练神经网络模型"""
        np.random.seed(5)
        for i in range(0,self.num_iterations):
            #正向传播，得到网络输出，以及每一层的参数信息
            (output, caches) = self.forward_propagation(self.x)
            # 计算网络输出误差
            cost = self.compute_error(output)
            # 反向传播，得到梯度信息
            grads = self.back_propagation(output,caches)
            #根据梯度信息，更新权重w和偏置b
            self.update_w_and_b(grads)
            # 当次迭代结束，打印误差信息
            if self.print_cost and i%1000 ==0:
                print("Cost after iteration %i:%f" % (i,cost))
            if self.print_cost and i%1000 == 0:
                self.costs.append(cost)
            #模型训练完后显示误差曲线
            if False:
                plt.plot(np.squeeze(self.costs))
                plt.ylabel(u'神经网络误差', fontproperties = font)
                plt.xlabel(u'迭代次数（*100）',fontproperties = font)
                plt.show()
        return self.w, self.b
    def predict_by_modle(self,x):
        """使用训练好的模型（即最后求得w,b参数）来决策输入的样本的结果"""
        output,_ = self.forward_propagation(x.T)
        output = output.T
        result = output/np.sum(output, axis=1, keepdims = True )
        return np.argmax(result, axis = 1)

def plot_decision_boundary(xy, colors, pred_func):
    # xy是坐标点的集合，把集合的范围算出来
    # 加减0.5相当于扩大画布的范围，不然画出来的图坐标点会落在图的边缘，处女座可选
    x_min,x_max = xy[:,0].min() - 0.5, xy[:,0].max() + 0.5
    y_min,y_max = xy[:,1].min() - 0.5, xy[:,1].max() + 0.5

    #以h为分辨率，生成采样点的网络，就像一张网覆盖所有颜色点
    h = .01
    xx, yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))# 步进为0.01

    #把网格点集合作为输入到模型，也就是预测这个采样点是什么颜色的点，从而得到一个决策面
    Z = pred_func(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    # 利用等高线，把预测的结果画出来，效果上就是画出红蓝点的分界线
    plt.contourf(xx,yy,Z,cmap = plt.cm.Spectral)
    # 训练用的红蓝点点也画出来
    plt.scatter(xy[:,0], xy[:,1],c = colors,marker = 'o',cmap = plt.cm.Spectral,edgecolors = 'black')

if __name__ == "__main__":
        plt.figure(figsize = (16,32))
        # 用sklearn的数据样本集，产生2种颜色的坐标点，noise是噪声系数，噪声越大，2种颜色的点分布越凌乱
        xy,colors = sklearn.datasets.make_moons(60, noise = 1.0)

        # 因为点的颜色是1bit,我们设计一个神经网络，输出层有两个神经元
        # 标定输出[1，0]为红色点，输出[0，1]为蓝色点
        expected_output = []
        for c in colors:
            if c == 1:
                expected_output.append([0,1])#这个意思就是添加了一个[0，1]的元素，即元素就是[0，1]
            else:
                expected_output.append([1,0])
        expected_output = np.array(expected_output).T

        # 设计3层网络，改变隐藏神经原的个数，观察神经网络分类红蓝点的效果
        hidden_layer_neuron_num_list = [1,2,4,10,20,50]
        for i,hidden_layer_neuron_num in enumerate(hidden_layer_neuron_num_list):
            plt.subplot(5,2,i+1)
            plt.title(u'隐藏神经元数量：%d' % hidden_layer_neuron_num, fontproperties = font)

            nn = NeuralNetwork([2,hidden_layer_neuron_num,2],True)

            # 输出和输入层都是两个节点，所以输入和输出的数据集合都要是 n*2的矩阵
            nn.set_xy(xy.T, expected_output)
            nn.set_num_iterations(30000)
            nn.set_learning_rate(0.1)
            (w, b) = nn.traning_model()
            plot_decision_boundary(xy,colors,nn.predict_by_modle)
        plt.show()