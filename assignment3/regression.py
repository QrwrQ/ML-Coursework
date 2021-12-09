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
#Normalized feature
train_feature=train_data.getFeature(SampleManager.ZSCORE)
validation_feature=validation_data.getFeature(SampleManager.ZSCORE)
test_feature=test_data.getFeature(SampleManager.ZSCORE)


#node (2,10,100),hidden_layer:(1,2,10)
for node_num in (2,10,100):
    for hidden_num in (1,2,10):
        hidden_info=[]
        for i in range(hidden_num):
            hidden_info.append(node_num)
        ann=ANN(8,1,hidden_info)
        ann.setActivationFunction([tf.nn.sigmoid])
        ann.setLossType(ANN.SQUARED_DIFFERENCE)
        ann.setHyperparameter(0.01,2500)
        ann.create(2333)
        train_info=ann.start(train_feature,train_data.getLabel(),1,validation_feature,validation_data.getLabel())
        test_info=ann.start(train_feature,train_data.getLabel(),1,test_feature,test_data.getLabel())
        pred=ann.predict(test_feature)
        plt.plot(train_info[1], 'r.-', train_info[2], 'b.-',test_info[2],'y.-')
        plt.title('node num: %d,layer num: %d'%(node_num,hidden_num))
        plt.legend(('train RMSE','validation RMSE','test RMSE'),loc='upper left')
        plt.show()
        plt.plot(test_data.getLabel(), 'ro', pred, 'bo')
        plt.legend(('true','pred'),loc='upper left')
        plt.show()




        
#create neural network and find optimal parameter
# (learning rate and epochs)
# optimal_parameter=None
loss_min=100
loss_min_train=0
loss_min_va=0
for i in (0.001,0.005,0.01,0.02,0.05,0.08,0.1,0.3,0.5):
    learning_rate=i
    for number_epochs in range(1000,5000,500):
        ann=ANN(8,1,[10,10])
        ann.setActivationFunction([tf.nn.sigmoid])
        ann.setLossType(ANN.SQUARED_DIFFERENCE)
        ann.setHyperparameter(learning_rate,number_epochs)
        ann.create(2333)
        #train
        train_info=ann.start(train_feature,train_data.getLabel(),500,validation_feature,validation_data.getLabel())
        #test loss
        pred=ann.predict(test_feature)
        loss=getRMSE(pred,test_data.getLabel())
        if loss<loss_min:
            loss_min=loss
            optimal_parameter=(learning_rate,number_epochs)
            loss_min_train=train_info[1][-1]
            loss_min_va=train_info[2][-1]
        print('learning_rate: %f'%learning_rate,'epochs: %d'%number_epochs,'RMSE: %f'%loss)
        print('loss_min: %f'%loss_min,'parameter:',optimal_parameter,'train_info:',loss_min_train,'|',loss_min_va)

ann=ANN(8,1,[10,10])
ann.setActivationFunction([tf.nn.sigmoid])
ann.setLossType(ANN.SQUARED_DIFFERENCE)
ann.setHyperparameter(0.001,2500)
ann.create(2333)
ann.start(train_feature,train_data.getLabel(),0,validation_feature,validation_data.getLabel())
pred=ann.predict(test_feature)
plt.plot(test_data.getLabel(), 'ro', pred, 'bo')
plt.legend(('true','pred'),loc='upper left')
plt.show()



#seed 2333 learning_rate=0.001 number_epochs=5000
learning_rate=0.001
number_epochs=5000
ann=ANN(8,1,[10,10])
ann.setActivationFunction([tf.nn.sigmoid])
ann.setLossType(ANN.SQUARED_DIFFERENCE)
ann.setHyperparameter(learning_rate,number_epochs)
ann.create(2333)
train_info=ann.start(train_feature,train_data.getLabel(),1,validation_feature,validation_data.getLabel())
test_info=ann.start(train_feature,train_data.getLabel(),1,test_feature,test_data.getLabel())
plt.plot(train_info[1], 'r.-', train_info[2], 'b.-',test_info[2],'y.-')
plt.legend(('train RMSE','validation RMSE','test RMSE'),loc='upper left')
plt.show()

# seed 2333 learning_rate=(0.001,0.01,0.05,1) number_epochs=2500
for learning_rate in (0.001,0.01,0.05,1):
    ann=ANN(8,1,[10,10])
    ann.setActivationFunction([tf.nn.sigmoid])
    ann.setLossType(ANN.SQUARED_DIFFERENCE)
    ann.setHyperparameter(learning_rate,2500)
    ann.create(2333)
    train_info=ann.start(train_feature,train_data.getLabel(),1,validation_feature,validation_data.getLabel())
    test_info=ann.start(train_feature,train_data.getLabel(),1,test_feature,test_data.getLabel())
    plt.plot(train_info[1], 'r.-', train_info[2], 'b.-',test_info[2],'y.-')
    plt.title('learning_rate: %f'%learning_rate)
    plt.legend(('train RMSE','validation RMSE','test RMSE'),loc='upper left')
    plt.show()



#10-flod
RMSE_list=[]
learning_rate=0.001
number_epochs=2500
ann=ANN(8,1,[10,10])
ann.setActivationFunction([tf.nn.sigmoid])
ann.setLossType(ANN.SQUARED_DIFFERENCE)
ann.setHyperparameter(learning_rate,number_epochs)
ann.create(2333)
for kfold_data in simple.kFold(10):
    kfold_train_data=kfold_data['train_data']
    kfold_test_data=kfold_data['test_data']
    train_info=ann.start(kfold_train_data.getFeature(SampleManager.ZSCORE),kfold_train_data.getLabel(),2500,kfold_test_data.getFeature(SampleManager.ZSCORE),kfold_test_data.getLabel())
    RMSE_list.append((train_info[1],train_info[2]))
print(RMSE_list)
train_RMSE_sum=0
test_RMSE_sum=0
for i in RMSE_list:
    train_RMSE_sum+=i[0][0]
    test_RMSE_sum+=i[1][0]
mean_train_RMSE=train_RMSE_sum/10
mean_test_RMSE=test_RMSE_sum/10
print('mean_train_RMSE:',mean_train_RMSE,'mean_train_RMSE:',mean_test_RMSE)
