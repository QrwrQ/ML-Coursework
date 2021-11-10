clc
clear all
close all

% 加载数据
load carsmall
rng 'default'  % For reproducibility

% 马力和重量作为自变量，MPG作为因变量
X = [Horsepower,Weight];
Y = MPG

% 返回一个默认的回归支持向量模型
Mdl = fitrsvm(X,Y)