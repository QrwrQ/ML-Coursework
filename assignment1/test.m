clc
clear all
close all
url="https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/"
% 加载数据
websave('abalone.csv',url);
% Tbl = readtable('forestfires.csv');
Tbl=csvread('forestfires.csv');
Tbl_x=Tbl(2:200,7:10);
Tbl_y=Tbl(2:200,11);
length(Tbl_y);
for i=1:length(Tbl_y)
%     if Tbl_y(i)
    if Tbl_y(i)>0
        Tbl_y(i)=1; 
    end
end
Y=nominal(Tbl_y);
% show the data distribution
% subplot(1,2,1)
% gscatter(Tbl_x(:,1),Tbl_x(:,2),Y,'rg','+*');

% Mdl=fitcsvm(Tbl_x,Y,'KernelFunction','linear')
Mdl=fitcsvm(Tbl_x,Y,'KernelFunction','Gaussian')

subplot(1,2,2)
h = nan(3,1);
h(1:2) = gscatter(Tbl_x(:,1),Tbl_x(:,2),Y,'rg','+*');
hold on
h(3) =plot(Tbl_x(Mdl.IsSupportVector,1),Tbl_x(Mdl.IsSupportVector,2), 'ko');

% w=-Mdl.Beta(1,1)/Mdl.Beta(2,1);%斜率
% b=-Mdl.Bias/Mdl.Beta(2,1);%截距
% x_ = 0:0.01:10;
% y_ = w*x_+b;
% plot(x_,y_)
% hold on
% legend(h,{<!-- -->'-1','+1','Support Vectors'},'Location','Southeast');
% axis equal
% hold off

% [lable,score]=predict(Mdl,Tbl_x)
% C = textscan(file_id, '%s%d%f%d', 'Delimiter', ',', 'HeaderLines', 1 );
% fclose(file_id);


% rng 'default'  % For reproducibility
% 
% % 马力和重量作为自变量，MPG作为因变量
% X = [Horsepower,Weight];
% Y = MPG;
% 
% % 返回一个默认的回归支持向量模型
% Mdl = fitrsvm(X,Y)
% 
% MdlStd = fitrsvm(X,Y,'Standardize',true)
% l = resubLoss(Mdl)
% lStd = resubLoss(MdlStd)


