function prediction=Re_DT_predict(Tree,X)
% tr=Tree;
prediction=[];
[r,c]=size(X);
for i=1:r
    tr=Tree;
    x_e=X(i,:);
    while isempty(tr.prediction)
        if x_e(tr.attribute)<tr.threshold
            tr=tr.kids{1};
        else
            tr=tr.kids{2};
        end
    end
    prediction=[prediction;tr.prediction];
end
end