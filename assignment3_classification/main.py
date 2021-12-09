import tensorflow as tf
import numpy as np
from ANN import *
import matplotlib.pyplot as plt
feature_index=4
# data=np.loadtxt('数据\\forestfires.csv',skiprows=1,usecols=range(4,13),delimiter=',')
data=np.loadtxt('iris.csv',delimiter=',')
feature=data[:,0:feature_index]
# label=np.log(data[:,feature_index]+1)
label=data[:,feature_index:]
simple=SampleManager(feature,label)
experimental_data=simple.getExperimentalData()
train_data=experimental_data['train_data']
validation_data=experimental_data['validation_data']
test_data=experimental_data['test_data']




#create neural network and find optimal parameter
# (learning rate and epochs)
optimal_parameter=None
loss2=0
for i in range(1,100,5):
    learning_rate=i/100
    for number_epochs in range(1000,10000,1000):
        ann=ANN(4,3,[10,10])
        ann.setActivationFunction([tf.nn.sigmoid],tf.nn.softmax)
        ann.setLossType(ANN.SOFTMAX_CROSS_ENTROPY)
        ann.setHyperparameter(learning_rate,number_epochs)
        ann.create()
        #train
        # train_featurn=train_data.getFeature(SampleManager.ZSCORE)
        train_featurn = train_data.getFeature()
        # validation_featurn=validation_data.getFeature(SampleManager.ZSCORE)
        validation_featurn = validation_data.getFeature()
        ac_rate=ann.start(train_featurn,train_data.getLabel(),10,validation_featurn,validation_data.getLabel())
        #test loss
        pred = ann.predict(test_data.getFeature())
        accuracy=getAccuracy(test_data.getLabel(),pred)
        # gap=(pred.max()-pred.min())/3
        # pred_min=pred.min()
        # for i in range(len(pred)):
        #     if pred[i]<pred_min+gap:
        #         pred[i]=1
        #     elif pred_min+gap<=pred[i]<=pred_min+2*gap:
        #         pred[i]=2
        #     else:
        #         pred[i]=3
        # loss = getAccuracy(train_featurn, train_data.getLabel())
        # plt.plot(get_typevalue(test_data.getLabel()),'ro', get_pre_value(pred), 'bo')
        # plt.ylabel('some numbers')
        plt.plot(ac_rate[1], 'r.-', ac_rate[2], 'b.-')
        plt.legend(('train accuracy', 'validation accuracy'), loc='upper left')
        plt.show()
        # accuracy = get_acurracy(pred,test_data.getLabel())
        print('accuracy=',accuracy)
        print("121")

        loss1=accuracy
        if loss1>loss2:
            loss2=loss1
            optimal_parameter=[i,number_epochs]
        input("Press Enter to continue...")

        #test



        # loss=ann.getAccuracy(ANN.REGRESSSION,test_data.getFeature(),test_data.getLabel())
        # loss1 = ann.getAccuracy(ANN.ANN.REGRESSSION, test_data.getFeature(), test_data.getLabel())

        # 评估模型得到最优参数
        optimal_parameter='optimal_parameter'



#10-flod
learning_rate=optimal_parameter[0]/100
number_epochs=optimal_parameter[1]
ann.setHyperparameter(learning_rate,number_epochs)
ann.create()
for kfold_data in simple.kFold(10):
    train_data=kfold_data['train']
    test_data=kfold_data['test']
    ann.start(train_data.getFeature(),train_data.getLabel())
    pred=ann.predict(test_data.getFeature())
    loss=ann.getAccuracy(ANN.REGRESSSION,test_data.getFeature(),test_data.getLabel())
