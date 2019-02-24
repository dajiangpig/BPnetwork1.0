from  BPActivationPart import *

class Load_Data:
    def __init__(self):
        """
        函数：初始化时自动执行
        功能：读数据
        输入：无
        输出：最原始的数据集，存给self结构体,self.data,self.label,self.size
        """
        self.data_path_data = 'D:/matlabqbit/Library/Library/2qubitPinshu.mat'
        self.data_path_label = 'D:/matlabqbit/Library/Library/2qubitLabel.mat'

        self.data = h5py.File(self.data_path_data)
        self.label = h5py.File(self.data_path_label)
        print('第一个mat文件中包含的目录：',self.data.keys())
        print('第二个mat文件中包含的目录：', self.label.keys())
        self.data = self.data['pinshu'][:]
        # data = [myfile[element[0]][:] for element in myfile['FEAT_Name']]
        self.label = [self.label[element[0]][:] for element in self.label['Y']]
        self.label = np.squeeze(self.label)
        self.size = self.label.shape[0]
        print('数据集：',self.data)
        print('正确答案：',self.label)
        print('数据集大小：',self.size)
        print('初始化成功')

    def pre_process_data(self):
        """
        函数：预处理，对输入数据进行简单的预处理
        功能：由于给出的是5个频数，要先进行预处理
        输入：type：选择类型，是平均还是随机
        输出：处理后的最终结果，存给self结构体,self.x
        用于测试的代码：
        for i in range(self.data.shape[0]): 这一部分写了用于测试
            print(i)
        print("首先看找对了没得：",self.data[:,0:5])
        print("axis0：",np.sum(self.data[:,0:5],axis = 0))
        print("axis1：",np.sum(self.data[:,0:5],axis = 1))
        """
        # data这里要求平均值，并且去掉nan异常数据
        self.data = np.array(self.data)
        self.x = [(np.sum(self.data[:,i*5:i*5+5],axis = 1)/5) for i in range(self.size)]
        self.x = np.squeeze(self.x)
        a = np.isnan(self.x)
        b = np.where(a)
        c = np.unique(b[0])
        self.x = np.delete(self.x,c,axis = 0)
        self.label = np.delete(self.label,c,axis = 0)
        e = np.isnan(self.x)
        f = np.where(e)

        #label这里要避免0、1出现，改为浮点数0.1，1.1
        self.label = self.label.astype(float)
        for i in range(np.size(self.label)):
            if self.label[i] == 1:
                self.label[i] = 0.9
            else:
                self.label[i] = 0.1
        print('预处理成功')

    def divide_training_and_test_set(self,rate = 0.5):
        """
        函数：划分训练集和测试集
        功能：划分训练集和测试集，按照你想要的比例
        输入：rate : 训练集所占总数据的比例,默认值是0.5
        输出：
        """
        self.train_input = self.x[0:int(rate * self.size),:]
        self.test_input = self.x[int(rate * self.size):self.size,:]
        self.train_label = self.label[0:int(rate * self.size)]
        self.test_label = self.label[int(rate*self.size):self.size]
        print('成功划分训练集和测试集')