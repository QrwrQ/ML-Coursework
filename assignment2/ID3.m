function [attribute,threshold,simple_cell]=ID3(features,labels,min_gain,min_num)
attribute=[];
threshold=[];
simple_cell=[];
gain=[];
[row,col]=size(features);
if row<=min_num 
    return;
end
label1=labels(1);
label2_local=find(labels~=label1);
if isempty(label2_local)
    return;
end
label2=labels(label2_local(1));
comentropy=getEntropy(labels,label1,label2);
for i=1:col
    simple=[features,labels];
    simple=sortrows(simple,i);
    features=simple(:,1:end-1);
    labels=simple(:,end);
    cut_point=[];
    gain_j=[];
    simple_cell_temp=[];
    for j=1:row-1
        if(features(j,i)==features((j+1),i))
            continue;
        end
        cut_point_temp=(features(j,i)+features((j+1),i))/2;
        c_entropy=j/row*getEntropy(labels(1:j),label1,label2)+(row-j)/row*getEntropy(labels(j+1:end),label1,label2);
        gain_temp=comentropy-c_entropy;
        if gain_temp<min_gain
            continue;
        end
        if isempty(cut_point)
            cut_point=cut_point_temp;
            gain_j=gain_temp;
            simple_cell_temp={features(1:j,:),labels(1:j,:);features(j+1:end,:),labels(j+1:end,:)};
        else
            if(gain_temp>gain_j)
                cut_point=cut_point_temp;
                gain_j=gain_temp;
                simple_cell_temp={features(1:j,:),labels(1:j,:);features(j+1:end,:),labels(j+1:end,:)};
            end
        end
    end
    if ~isempty(cut_point)
        if isempty(attribute)
            attribute=i;
            threshold=cut_point;
            gain=gain_j;
            simple_cell=simple_cell_temp;
        else
            if gain<gain_j
                attribute=i;
                threshold=cut_point;
                gain=gain_j;
                simple_cell=simple_cell_temp;
            end
        end
    end
end
end