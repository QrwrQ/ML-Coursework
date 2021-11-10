clc
clear all
close all

% Tbl = readtable('forestfires.csv');
Tbl=csvread('forestfires.csv');
Tbl_x=Tbl(2:400,5:10);
Tbl_t_x=Tbl(402:500,5:10);
Tbl_t_y=Tbl(402:500,11);
Tbl_y=Tbl(2:400,11);
length(Tbl_y);
for i=1:length(Tbl_t_y)
%     if Tbl_y(i)
    if Tbl_t_y(i)>0
        Tbl_t_y(i)=1; 
    end
end
for i=1:length(Tbl_y)
%     if Tbl_y(i)
    if Tbl_y(i)>0
        Tbl_y(i)=1; 
    end
end

Y=nominal(Tbl_y);
Tbl_t_y=nominal(Tbl_t_y)
% show the data distribution
% subplot(1,2,1)
% gscatter(Tbl_x(:,1),Tbl_x(:,2),Y,'rg','+*');

 %Mdl=fitcsvm(Tbl_x,Y,'KernelFunction','linear','BoxConstraint',1)
%Mdl=fitcsvm(Tbl_x,Y,'KernelFunction','Gaussian','KernelScale',4,'BoxConstraint',0.1)
% Mdl=fitcsvm(Tbl_x,Y,'KernelFunction','polynomial','PolynomialOrder',9)
%Mdl = fitrsvm(Tbl_x,Tbl_y);
ParaAnalyse('C',Tbl_x,Y)

% vec_num=length(Mdl.SupportVectors)
% p_y=predict(Mdl,Tbl_t_x);
% accuracy = sum(predict(Mdl,Tbl_t_x) == Tbl_t_y)/length(Tbl_t_y)*100

function parame_an=ParaAnalyse(para,Tbl_x,Y)
    if para=='C'
        para_C=1:1:100;
        su_ve=[];
        for pa_v=para_C
            Mdl=fitcsvm(Tbl_x,Y,'KernelFunction','linear','BoxConstraint',pa_v);
            vec_num=length(Mdl.SupportVectors);
            su_ve(end+1)=vec_num;
            pa_v
        end
        plot(para_C,su_ve)
        
    end

end



% subplot(1,2,2)
% h = nan(3,1);
% h(1:2) = gscatter(Tbl_x(:,1),Tbl_x(:,2),Y,'rg','+*');
% hold on
% h(3) =plot(Tbl_x(Mdl.IsSupportVector,1),Tbl_x(Mdl.IsSupportVector,2), 'ko');

% w=-Mdl.Beta(1,1)/Mdl.Beta(2,1);%斜率
% b=-Mdl.Bias/Mdl.Beta(2,1);%截距
% x_ = 0:0.01:10;
% y_ = w*x_+b;
% plot(x_,y_)
% hold on
% legend(h,{<!-- -->'-1','+1','Support Vectors'},'Location','Southeast');
% axis equal
% hold off


