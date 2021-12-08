import tensorflow as tf
import numpy as np
from ANN import *
import matplotlib.pyplot as plt

data=np.loadtxt('数据\\forestfires.csv',skiprows=1,usecols=range(4,13),delimiter=',')
feature=data[:,0:8]
label=np.log(data[:,8]+1)
simple=SampleManager(feature,label.T)
experimental_data=simple.getExperimentalData()
train_data=experimental_data['train_data']
validation_data=experimental_data['validation_data']
test_data=experimental_data['test_data']




#create neural network and find optimal parameter
# (learning rate and epochs)
optimal_parameter=None
for i in range(1,100):
    learning_rate=i/100
    for number_epochs in range(1000,10000,100):
        ann=ANN(8,1,[10,10])
        ann.setActivationFunction([tf.nn.sigmoid,tf.nn.tanh],tf.nn.relu)
        ann.setLossType(ANN.SQUARED_DIFFERENCE)
        ann.setHyperparameter(learning_rate,number_epochs)
        ann.create()
        #train
        train_featurn=train_data.getFeature(SampleManager.ZSCORE)
        validation_featurn=validation_data.getFeature(SampleManager.ZSCORE)
        ann.start(train_featurn,train_data.getLabel(),10,validation_featurn,validation_data.getLabel())
        #test loss
        loss=ann.getAccuracy(ANN.REGRESSSION,train_featurn,train_data.getLabel())
        pred=ann.predict(train_featurn)
        plt.plot(train_data.getLabel(), 'ro', pred, 'bo')
        plt.ylabel('some numbers')
        plt.show()
        #test
        pred=ann.predict(test_data.getFeature())
        loss=ann.getAccuracy(ANN.REGRESSSION,test_data.getFeature(),test_data.getLabel())
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
