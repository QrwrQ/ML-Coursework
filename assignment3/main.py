import tensorflow as tf
import numpy as np
from ANN import *
import matplotlib.pyplot as plt

data=np.loadtxt('F:\\Msc诺丁汉\\秋季\\机器学习\\cw3\\python代码\\数据\\forestfires.csv',skiprows=1,usecols=range(4,13),delimiter=',')
feature=data[:,0:8]
label=np.log(data[:,8]+1)
simple=SampleManager(feature,label.T)
experimental_data=simple.getExperimentalData()
train_data=experimental_data['train_data']
validation_data=experimental_data['validation_data']
test_data=experimental_data['test_data']
#特征值标准化
train_feature=train_data.getFeature(SampleManager.ZSCORE)
validation_feature=validation_data.getFeature(SampleManager.ZSCORE)
test_feature=test_data.getFeature(SampleManager.ZSCORE)


# 研究隐藏层对模型的影响
#隐藏层从1-3，节点数2, 5，10，20
#其他变量不变
# print_info=[]
# for node_num in (2,5,10,20):
#     for hidden_num in range(1,4):
#         hidden_info=[]
#         for i in range(hidden_num):
#             hidden_info.append(node_num)
#         ann=ANN(8,1,hidden_info)
#         ann.setActivationFunction([tf.nn.sigmoid])
#         ann.setLossType(ANN.SQUARED_DIFFERENCE)
#         ann.setHyperparameter(0.1,5000)
#         ann.create()
#         train_info=ann.start(train_feature,train_data.getLabel(),10,validation_feature,validation_data.getLabel())
#         pred=ann.predict(test_feature)
#         del(ann)
#         rmse=getRMSE(pred,test_data.getLabel())
#         print_info.append((node_num,hidden_num,rmse))
# for i in range(len(print_info)):
#     print('node num: %d'%print_info[i][0],'hidden num: %d'%print_info[i][1],'rmse: %f'%print_info[i][2])


        
#create neural network and find optimal parameter
# (learning rate and epochs)
optimal_parameter=None
for i in range(1,500):
    learning_rate=i/1000
    for number_epochs in range(1000,10000,100):
        ann=ANN(8,1,[10,10])
        ann.setActivationFunction([tf.nn.sigmoid])
        ann.setLossType(ANN.SQUARED_DIFFERENCE)
        ann.setHyperparameter(learning_rate,number_epochs)
        ann.create(666)
        #train
        ann.start(train_feature,train_data.getLabel(),0,validation_feature,validation_data.getLabel())
        #test loss
        # pred=ann.predict(train_featurn)
        # plt.plot(train_data.getLabel(), 'ro', pred, 'bo')
        # plt.ylabel('some numbers')
        # plt.show()
        #test
        pred=ann.predict(test_feature)
        loss=getRMSE(pred,test_data.getLabel())
        print('learning_rate: %f'%learning_rate,'epochs: %d'%number_epochs,'RMSE: %f'%loss)
        pass
# 评估模型得到最优参数
optimal_parameter='optimal_parameter'



#10-flod
learning_rate='optimal parameter'
number_epochs='optimal parameter'
ann.setHyperparameter(learning_rate,number_epochs)
ann.create()
for kfold_data in simple.kFold(10):
    train_data=kfold_data['train']
    test_data=kfold_data['test']
    ann.start(train_data.getFeature(),train_data.getLabel())
    pred=ann.predict(test_data.getFeature())
    loss=ann.getAccuracy(ANN.REGRESSSION,test_data.getFeature(),test_data.getLabel())
