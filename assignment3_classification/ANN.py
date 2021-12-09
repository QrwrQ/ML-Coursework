import numpy as np
import tensorflow as tf
import math
import logging
import random


logging.basicConfig(level=logging.INFO)

class ANN:
    __activationFunction=None
    __output_layer_active_function=None
    __loss_type=None
    __optimizer=None
    __X=None
    __Y=None
    __learning_rate=None
    __number_epochs=None
    __batch_size=None
    __neural_network=None
    __sess=None

    REGRESSSION=1
    CLASS=2

    SQUARED_DIFFERENCE=1
    SOFTMAX_CROSS_ENTROPY=2


    def __init__(self,input_num,output_num,hidden_info):
        self.__input_num=input_num
        self.__output_num=output_num
        self.__hidden_num=hidden_info

    def setActivationFunction(self,activation_function_list,output_layer_active_function=None):
        if len(activation_function_list)==len(self.__hidden_num) or len(activation_function_list)==1:
            self.__activationFunction=activation_function_list
            self.__output_layer_active_function=output_layer_active_function
        else:
            raise Exception("Error shape!")

    def setLossType(self,loss_type):
        self.__loss_type=loss_type

    def setHyperparameter(self,learning_rate,number_epochs,batch_size=0):
        self.__learning_rate=learning_rate
        self.__number_epochs=number_epochs
        self.__batch_size=batch_size

    def __isValid(self):
        if self.__activationFunction==None or self.__loss_type not in (ANN.SQUARED_DIFFERENCE,ANN.SOFTMAX_CROSS_ENTROPY) or self.__learning_rate==None or self.__number_epochs==None or self.__batch_size==None:
            return False
        else:
            return True
    # create notwork
    # def create(self,seed=None):
    #     if not self.__isValid():
    #         raise Exception('No set!')
    #     # if seed != None:
    #     #     random.seed(seed)
    #     input_num=self.__input_num
    #     output_num=self.__output_num
    #     hidden_info=self.__hidden_num
    #     hidden_num=len(hidden_info)
    #     activationFunction=[]
    #     if len(self.__activationFunction)==1:
    #         for i in range(hidden_num):
    #             activationFunction.append(self.__activationFunction[0])
    #     else:
    #         activationFunction=self.__activationFunction
    #
    #     X=tf.placeholder(tf.float32, [None, input_num])
    #     Y=tf.placeholder(tf.float32, [None, output_num])
    #
    #     self.__X=X
    #     self.__Y=Y
    #
    #     layer_last=None
    #     for i in range(hidden_num):
    #         b=tf.Variable(tf.random_normal([hidden_info[i]]))
    #         if i==0:
    #             w=tf.Variable(tf.random_normal([input_num, hidden_info[i]]))
    #             layer_last=activationFunction[i](tf.add(tf.matmul(X, w), b))
    #         else:
    #             w=tf.Variable(tf.random_normal([hidden_info[i-1], hidden_info[i]]))
    #             layer_last=activationFunction[i](tf.add(tf.matmul(layer_last, w), b))
    #
    #     b=tf.Variable(tf.random_normal([output_num]))
    #     w=tf.Variable(tf.random_normal([hidden_info[-1], output_num]))
    #     if self.__output_layer_active_function==None:
    #         self.__neural_network=tf.add(tf.matmul(layer_last, w),b)
    #     else:
    #         self.__neural_network=self.__output_layer_active_function(tf.add(tf.matmul(layer_last, w),b))
    #
    #     if self.__loss_type==ANN.SQUARED_DIFFERENCE:
    #         loss_op=tf.reduce_mean(tf.math.squared_difference(self.__neural_network,Y))
    #     elif self.__loss_type==ANN.SOFTMAX_CROSS_ENTROPY:
    #         loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.__neural_network,labels=Y))
    #     else:
    #         raise Exception('error loss type')
    #     self.__optimizer=tf.train.GradientDescentOptimizer(self.__learning_rate).minimize(loss_op)

    # start train
    def create(self,seed=None):
        if not self.__isValid():
            raise Exception('No set!')
        if seed!=None:
            random.seed(seed)
        input_num=self.__input_num
        output_num=self.__output_num
        hidden_info=self.__hidden_num
        hidden_num=len(hidden_info)
        activationFunction=[]
        if len(self.__activationFunction)==1:
            for i in range(hidden_num):
                activationFunction.append(self.__activationFunction[0])
        else:
            activationFunction=self.__activationFunction

        X=tf.placeholder(tf.float32, [None, input_num])
        Y=tf.placeholder(tf.float32, [None, output_num])

        self.__X=X
        self.__Y=Y

        layer_last=None
        for i in range(hidden_num):
            b=tf.Variable(tf.random_normal([hidden_info[i]],seed=random.randint(-1000,1000)))
            if i==0:
                w=tf.Variable(tf.random_normal([input_num, hidden_info[i]],seed=random.randint(-1000,1000)))
                layer_last=activationFunction[i](tf.add(tf.matmul(X, w), b))
            else:
                w=tf.Variable(tf.random_normal([hidden_info[i-1], hidden_info[i]],seed=random.randint(-1000,1000)))
                layer_last=activationFunction[i](tf.add(tf.matmul(layer_last, w), b))

        b=tf.Variable(tf.random_normal([output_num],seed=random.randint(-1000,1000)))
        w=tf.Variable(tf.random_normal([hidden_info[-1], output_num],seed=random.randint(-1000,1000)))
        if self.__output_layer_active_function==None:
            self.__neural_network=tf.add(tf.matmul(layer_last, w),b)
        else:
            self.__neural_network=self.__output_layer_active_function(tf.add(tf.matmul(layer_last, w),b))

        if self.__loss_type==ANN.SQUARED_DIFFERENCE:
            loss_op=tf.reduce_mean(tf.math.squared_difference(self.__neural_network,Y))
        elif self.__loss_type==ANN.SOFTMAX_CROSS_ENTROPY:
            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.__neural_network,labels=Y))
        else:
            raise Exception('error loss type')
        self.__optimizer=tf.train.GradientDescentOptimizer(self.__learning_rate).minimize(loss_op)



    def start(self,feature,label,interval=0,validation_feature=None,validation_label=None):
        if self.__optimizer==None:
            raise Exception('No neural network!')
        feature=self.__errorShape(feature)
        label=self.__errorShape(label)

        #save info
        train_accuracy_info=[]
        validation_accuracy_info=[]
        epoch_info=[]

        if validation_feature is None or validation_label is None:
            validation_feature=feature
            validation_label=label
        else:
            if validation_feature.shape[0]!=validation_label.shape[0]:
                raise Exception('error shape!')
        # session
        if self.__sess!=None:
            self.__sess.close()
            self.__sess=None
        self.__sess=tf.Session()
        self.__sess.run(tf.global_variables_initializer())

        for epoch in range(self.__number_epochs):
            self.__sess.run(self.__optimizer,feed_dict={self.__X:feature,self.__Y:label})
            # print train info
            if interval>0:
                if (epoch+1) %interval==0:
                    if self.__loss_type==ANN.SQUARED_DIFFERENCE:
                        pred=self.__neural_network.eval({self.__X:feature},session=self.__sess)
                        train_accuracy=getRMSE(label,pred)
                        pred=self.__neural_network.eval({self.__X:validation_feature},session=self.__sess)
                        validation_accuracy=getRMSE(validation_label,pred)
                    elif self.__loss_type==ANN.SOFTMAX_CROSS_ENTROPY:
                        pred=self.__neural_network.eval({self.__X:feature},session=self.__sess)
                        train_accuracy=getAccuracy(label,pred)
                        pred=self.__neural_network.eval({self.__X:validation_feature},session=self.__sess)
                        validation_accuracy=getAccuracy(validation_label,pred)
                    print('epoch:','%d'%(epoch+1),'| train_accuracy:','%f'%(train_accuracy),'| validation_accuracy','%f'%(validation_accuracy))
                    epoch_info.append(epoch+1)
                    train_accuracy_info.append(train_accuracy)
                    validation_accuracy_info.append(validation_accuracy)
        train_info=(epoch_info,train_accuracy_info,validation_accuracy_info)
        return train_info


    def __errorShape(self,np_):
        if (len(np_.shape)==1):
            np_.shape=(np_.shape[0],1)
        return np_

    def predict(self,feature):
        if self.__sess==None:
            raise Exception("No model!")
        pred=self.__neural_network.eval({self.__X:feature},session=self.__sess)
        return pred

    def getAccuracy(self,type,feature,label):
        if type==ANN.REGRESSSION:
            accuracy=tf.keras.losses.MSE(self.__Y,self.__neural_network)
            return accuracy.eval({self.__X: feature, self.__Y: label},session=self.__sess)
        elif type==ANN.CLASS:
            
            correct_prediction1 = tf.equal(tf.argmax(self.__neural_network, 1),label)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
            return accuracy.eval({self.__X: feature},session=self.__sess)
        else:
            raise Exception('error type')


    def __del__(self):
        if self.__sess!=None:
            self.__sess.close()

def getRMSE(y1,y2):
    if len(y1)!=len(y2):
        raise Exception("error shape")
    sum=0
    for i in range(len(y1)):
        sum+=math.sqrt((y1[i]-y2[i])**2)
    return sum/len(y1)

def getAccuracy(y1,y2):
    if y1.shape!=y2.shape:
        raise Exception("error shape")
    right_num=0
    for i in range(y1.shape[0]):
        if y1[i,:].argmax()==y2[i,:].argmax():
            right_num+=1
    return right_num/y1.shape[0]


class SampleManager:

    __feature=None
    __label=None

    z_score_para=None
    min_max_para=None

    ZSCORE=0

    def __init__(self,feature,label):
        if feature.shape[0]!=label.shape[0]:
            raise Exception('error shape!')
        self.__feature=self.__errorShape(feature)
        self.__label=self.__errorShape(label)

    def __errorShape(self,np_):
        if (len(np_.shape)==1):
            np_.shape=(np_.shape[0],1)
        return np_

    def kFold(self,K):
        if K>self.__label.shape[0]:
            raise Exception('Cannot flod!')
        label=self.__label
        feature=self.__feature
        last_ptr=0
        num=math.ceil(label.shape[0]/K)
        for i in range(K):
            if last_ptr+num<=label.shape[0]:
                kdata_label_test=label[last_ptr:last_ptr+num,:]
                kdata_feature_test=feature[last_ptr:last_ptr+num,:]
                kdata_label_train=np.delete(label,np.s_[last_ptr:last_ptr+num],0)
                kdata_feature_train=np.delete(feature,np.s_[last_ptr:last_ptr+num],0)
                last_ptr+=num
            else:
                kdata_label_test=label[last_ptr:,:]
                kdata_feature_test=feature[last_ptr:,:]
                kdata_label_train=np.delete(label,np.s_[last_ptr:],0)
                kdata_feature_train=np.delete(feature,np.s_[last_ptr:],0)
            train_data=SampleManager(kdata_feature_train,kdata_label_train)
            test_data=SampleManager(kdata_feature_test,kdata_label_test)
            # get normalization para
            train_data.calZscorePara()
            test_data.z_score_para=train_data.z_score_para

            yield {'train_data':train_data,'test_data':test_data}

    # train:validation:test=6:2:2
    def getExperimentalData(self):
        label=self.__label
        feature=self.__feature
        num=math.ceil(label.shape[0]/10)
        train_data=SampleManager(feature[0:num*6,:],label[0:num*6,:])
        validation_data=SampleManager(feature[num*6:num*8,:],label[num*6:num*8,:])
        test_data=SampleManager(feature[num*8:,:],label[num*8:,:])
        # get normalization para
        train_data.calZscorePara()
        validation_data.z_score_para=train_data.z_score_para
        test_data.z_score_para=train_data.z_score_para
        return {'train_data':train_data,'validation_data':validation_data,'test_data':test_data}

    def getLabel(self):
        return self.__label
    def get_class_lable(self):
        for i in range(len(self.__label)):
            if self.__label[i]==1:
                self.__label[i]=[1,0,0]
            if self.__label[i]==2:
                self.__label[i]=[0,1,0]
            if self.__label[i]==3:
                self.__label[i]=[0,0,1]





    def calZscorePara(self):
        feature=self.__feature
        para=[]
        for i in range(feature.shape[1]):
            f=feature[:,i]
            u=f.mean()
            s=f.std()
            para.append((u,s))
        self.z_score_para=tuple(para)

    def zScore(self,recal=False):
        feature=self.__feature
        norm_feature=None
        if self.z_score_para==None or recal==True:
            para=[]
            for i in range(feature.shape[1]):
                f=feature[:,i]
                u=f.mean()
                s=f.std()
                f=(f-u)/s
                para.append((u,s))
                if norm_feature is None:
                    norm_feature=f
                else:
                    norm_feature=np.column_stack((norm_feature,f))
            self.z_score_para=tuple(para)
        else:
            for i in range(feature.shape[1]):
                f=feature[:,i]
                u=self.z_score_para[i][0]
                s=self.z_score_para[i][1]
                f=(f-u)/s
                if norm_feature is None:
                    norm_feature=f
                else:
                    norm_feature=np.column_stack((norm_feature,f))
        return norm_feature

            

    def minMax(feature,recal=False):
        pass

    def getFeature(self,normalization=None):
        if normalization==None:
            return self.__feature
        elif normalization==SampleManager.ZSCORE:
            return self.zScore()


if __name__=='__main__':
    import matplotlib.pyplot as plt
    #读取数据
    data=np.loadtxt('数据\\forestfires.csv',skiprows=1,usecols=range(4,13),delimiter=',')
    feature=data[:,0:8]
    label=np.log(data[:,8]+1)
    # label=data[:,8]
    simple=SampleManager(feature,label.T)

    #划分训练集验证集和测试集
    experimental_data=simple.getExperimentalData()
    train_data=experimental_data['train_data']
    validation_data=experimental_data['validation_data']
    test_data=experimental_data['test_data']

    #创建ANN
    ann=ANN(8,1,[15,15])#自定义输入、输出、隐藏层
    ann.setActivationFunction([tf.nn.tanh,tf.nn.sigmoid])#自定义激活函数
    ann.setLossType(ANN.SQUARED_DIFFERENCE)# ANN.SQUARED_DIFFERENCE适用于回归 ANN.SOFTMAX_CROSS_ENTROPY适用于分类
    ann.setHyperparameter(0.01,5000)#自定义learn_rate 和 number_epochs
    ann.create(66)

    #训练
    #得到标准化后特征（也可不用标准化，看模型效果决定是否标准化）SampleManager.ZSCORE
    train_featurn=train_data.getFeature(SampleManager.ZSCORE)
    validation_feature=validation_data.getFeature(SampleManager.ZSCORE)
    test_feature=test_data.getFeature(SampleManager.ZSCORE)
    #开始训练模型
    train_info=ann.start(train_featurn,train_data.getLabel(),10,validation_feature,validation_data.getLabel())
    #将训练信息可视化
    plt.plot(train_info[1], 'r.-', train_info[2], 'b.-')
    plt.legend(('train accuracy','validation accuracy'),loc='upper left')
    plt.show()

    #查看该模型在测试集上的表现
    pred=ann.predict(test_feature)
    print(getRMSE(pred,test_data.getLabel()))
    plt.plot(test_data.getLabel(), 'ro', pred, 'bo')
    plt.ylabel('some numbers')
    plt.show()

def get_acurracy(pred,label):
    print('rrr')
    k=0
    for i in range(len(pred)):
        if pred[i]==label[i]:
            k=+1
    return k/len(pred)

def get_typevalue(data):
    type_list=[]
    for i in range(len(data)):
        if data[i,0]==1:
            type_list.append(1)
        if data[i,1]==2:
            type_list.append(2)
        if data[i,2]==3:
            type_list.append(3)
    return type_list

def get_pre_value(data):
    type_list=[]
    for i in range(len(data)):
        ty=np.where(data[i] == np.max(data[i], axis=0))
        type_list.append(ty)
    return type_list


