clear all;
clc;

filepath="";
% data1=importdata(filepath+"Iris-setosa.txt");
% data2=importdata(filepath+"Iris-versicolor.txt");
% sample=[data1(1:40,:);data2(1:40,:)];
% label=[ones(40,1);-ones(40,1)];
% Mdl=fitcsvm(sample,label, 'KernelFunction','linear', 'BoxConstraint',1);
% predict_label=predict(Mdl,[data1(41:end,:);data2(41:end,:)])


% data pretreatment
data3=importdata(filepath+"forestfires_or.csv",',',1)
sample=data3.data(:,1:end-1);
label=log(data3.data(:,end)+1);
y_label=Labelling(label)
% for i=1:k
%     Mdl=fitcsvm(sample(1:510,5:end),y_label(1:510,:),'KernelFunction','gaussian','Epsilon',0.33,'Standardize',true,'BoxConstraint',3);
%
% end
% predict_label=exp(predict(Mdl,sample(511:end,5:end)))-1
% a=crossval(Mdl);



%inner 分类
in_k=2
k=10;
in_kdata=KData(in_k,sample,y_label)
kdata=KData(k,sample,y_label);

%Gaussian RBF：C 'KernelScale','Epsilon'
mdl_in_cell=cell(in_k,1)
mdl_cell=cell(k,1);

%creat the parameter combination
ker_li=1:1:10;
ep_li=1:1:10;
para_matrx={ker_li;ep_li};
h_para=combination(para_matrx);
accuracy=0;
op_pa=zeros(1,2);
%inner_cross
for i=1:in_k
    data_in_train=in_kdata;
    data_in_train(i,:)=[];
    data_in_cheak=in_kdata(i,:);
    for j=1:length(h_para)
        mdl_in_cell{i}=fitcsvm(cell2mat(data_in_train(:,1)),cell2mat(data_in_train(:,2)),"KernelFunction","rbf","KernelScale",h_para(j,1),"BoxConstraint",h_para(j,2));
        h_para(j,:)
        accuracy1 = sum(predict(mdl_in_cell{i},cell2mat(data_in_cheak(:,1))) == cell2mat(data_in_cheak(:,2)))/length(cell2mat(data_in_cheak(:,2)))*100

        if accuracy==accuracy1
            if((op_pa(1)+op_pa(1)>(h_para(j,1)+h_para(j,2))))
                op_pa=h_para(j,:);
            end
        end

        if accuracy1>accuracy
            accuracy=accuracy1
            op_pa=h_para(j,:)
        end
        
    end
end
%10fold cross validation
for i=1:k
    e=0.1+0.01*i;
    data_train=kdata;
    data_train(i,:)=[];
    data_cheak=kdata(i,:);
    mdl_cell{i}=fitcsvm(cell2mat(data_train(:,1)),cell2mat(data_train(:,2)),"KernelFunction","rbf","KernelScale",op_pa(1),"BoxConstraint",op_pa(2));
    accuracy = sum(predict(mdl_cell{i},cell2mat(data_cheak(:,1))) == cell2mat(data_cheak(:,2)))/length(cell2mat(data_cheak(:,2)))*100
end
% KData(5,sample,label);
combination({[1,2,3];[4,9];[5,6,7]})


%kadta{sample1,label1;sample2,label2;...}
%用于样本分块
function kdata=KData(k,sample,label)
[row_size,~]=size(sample);
n=fix(row_size/k);
kdata=cell(k,2);
for i=1:k
    kdata{i,1}=sample((i-1)*n+1:i*n,:);
    kdata{i,2}=label((i-1)*n+1:i*n,:);
end
if k*n<row_size
    for i=k*n+1:row_size
        kdata{i-k*n,1}=[kdata{i-k*n,1};sample(i,:)];
        kdata{i-k*n,2}=[kdata{i-k*n,2};label(i,:)];
    end
end
end

% 用于参数网格化搜索
function com=combination(range_cell)
row_size=size(range_cell,1);
times=1;
for i=1:row_size
    times=times*size(range_cell{i},2);
end
com=zeros(times,row_size);
for i=1:row_size
    if i==row_size
        last_double=range_cell{i};
        num_last_double=size(last_double,2);
        n=times/num_last_double;
        for j=1:n
            com((j-1)*num_last_double+1:j*num_last_double,end)=last_double';
        end
    else
        temp_double=range_cell{i};
        num_temp_double=size(temp_double,2);
        n=1;
        for j=i+1:row_size
            n=n*size(range_cell{j},2);
        end
        temp_term=(temp_double'*ones(1,n))';
        term=ones(size(temp_term,1)*size(temp_term,2),1);
        for k=1:size(temp_term,2)
            term((k-1)*size(temp_term,1)+1:k*size(temp_term,1),1)=temp_term(:,k);
        end
        num_term=size(term,1)*size(term,2);
        for k=1:times/num_term
            com((k-1)*num_term+1:k*num_term,i)=term(:,1);
        end
    end

end
end

function lb=Labelling(Y_li)
for i=1:length(Y_li)
    if Y_li(i)>0
        Y_li(i)=1;
    end
end
lb=Y_li
end
% name_value_range_cell形式：
% {name，value的取值范围}

