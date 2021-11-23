function Re_Node=CreateTree_re(tran_data_X,tran_data_Y,corefunc,layer,arg,min_num,k)
node=struct('op','','kids',[],'prediction',[],'attribute',[],'threshold',[]);
if k<layer
    Re_Node=node;
    [attribute,threshold,kids]=corefunc(tran_data_X,tran_data_Y,arg,min_num);
    if isempty(kids)
        Re_Node=node;
        r_pre=0;
        for i=1:length(tran_data_Y)
            r_pre=r_pre+tran_data_Y(i);
        end
        r_pre=r_pre/length(tran_data_Y);
        Re_Node.prediction=r_pre;
        return
    end
    Re_Node.op=num2str(attribute);
    Re_Node.attribute=attribute;
    Re_Node.threshold=threshold;
    left_X=kids{1,1};
    right_X=kids{2,1};
    left_Y=kids{1,2};
    right_Y=kids{2,2};
% left tree
    Re_Node.kids=cell(1,2);
    if length(left_Y)<min_num
        Re_Node.kids{1}=node;
        k_pre=0;
        for i=1:length(left_Y)
            k_pre=k_pre+left_Y(i);
        end
        k_pre=k_pre/length(left_Y);
        Re_Node.kids{1}.prediction=k_pre;
    else 
        Re_Node.kids{1}=CreateTree_re(left_X,left_Y,corefunc,layer,arg,min_num,k+1);
    end
%right
    if length(right_Y)<min_num
        Re_Node.kids{2}=node;
        k_pre=0;
        for i=1:length(right_Y)
            k_pre=k_pre+right_Y(i);
        end
        k_pre=k_pre/length(right_Y);
        Re_Node.kids{2}.prediction=k_pre;
    else 
        Re_Node.kids{2}=CreateTree_re(right_X,right_Y,corefunc,layer,arg,min_num,k+1);
    end
else
    Re_Node=node;
    r_pre=0;
    for i=1:length(tran_data_Y)
        r_pre=r_pre+tran_data_Y(i);
    end
    r_pre=r_pre/length(tran_data_Y);
    Re_Node.prediction=r_pre;
end