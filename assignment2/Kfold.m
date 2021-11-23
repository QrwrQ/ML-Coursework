function op_tree=Kfold(k,sample,label,corefunction)
kdata=KData(k,sample,label);
rm_max=0;
op_tree=[];
% predict_label=label*0;
for i=1:k
    data_train_cell=kdata;
    data_train_cell(i,:)=[];
    data_cheak_cell=kdata(i,:);
    tree=DecisionTree(cell2mat(data_train_cell(:,1)),cell2mat(data_train_cell(:,2)),'corefunction',corefunction);
%     DrawDecisionTree(tree);
    l=length(cell2mat(data_cheak_cell(:,1)));
    prediction=Re_DT_predict(tree,cell2mat(data_cheak_cell(:,1)));
    rmse=RMSE(cell2mat(data_cheak_cell(:,2)),prediction);
    if rmse>rm_max
        rm_max=rmse;
        op_tree=tree;
    end
    
end
% rmse=RMSE(label,predict_label);
end