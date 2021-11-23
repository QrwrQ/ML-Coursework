clear all;
clc;
filepath="";


data3=importdata(filepath+"forestfires_or.csv",',',1);
features=data3.data(:,1:end-1);
label=log(data3.data(:,end)+1);
op_tree=Kfold(10,features,label,'CART');
DrawDecisionTree(op_tree);
% tree=DecisionTree(features,label,'corefunction','CART');
% DrawDecisionTree(tree,'test');
% prd=Re_DT_predict(tree,features(1:20,:))

% 
% data1=importdata(filepath+"Iris-setosa.txt");
% data2=importdata(filepath+"Iris-versicolor.txt");
% features=[data1(1:40,:);data2(1:40,:)];
% label=[ones(40,1);-ones(40,1)];
% tree=DecisionTree(features,label,'corefunction','ID3');
% DrawDecisionTree(tree,'test');