import BPLoaddataPart as BP
import BPTrainPart as TI
import numpy as np

if __name__ == "__main__":
    # 首先加载所需处理的数据
    load_data = BP.Load_Data()
    # 其次对频数数据进行预处理（Matlab之前给的数据有一定缺陷）
    load_data.pre_process_data()
    #划分训练集和测试集，rate = 0.5，默认 为0.5
    load_data.divide_training_and_test_set()

    #下面是训练部分，首先把训练集输入到训练模块儿中
    #print(help(TI.Train_Data))
    print('下面开始训练啦！！！')
    train_data = TI.Train_Data()
    for j in range(200):
      for i in range(0,25000,8):
        train_data.change_dataset(load_data.train_input[i:i+8],load_data.train_label[i:i+8])
        for count in range(0,3):
            output,caches = train_data.forward_propagation()
            grads = train_data.backward_propagation(caches)
            train_data.update_w_and_b(grads)
            loss,_= train_data.compute_loss(caches[train_data.layers_num-1][0])
      print('当前迭代次数：',j)
      train_data.change_dataset(load_data.train_input, load_data.train_label)
      output, caches = train_data.forward_propagation()
      loss, _ = train_data.compute_loss(caches[train_data.layers_num - 1][0])
      print(loss)

    ##下面是测试部分
    print('下面开始测试啦！')
    train_data.change_dataset(load_data.test_input,load_data.test_label)
    output, caches = train_data.forward_propagation()
    loss,_ = train_data.compute_loss(caches[train_data.layers_num-1][0])
    print(loss)