% get tree.attribute and tree.threshold
function [attribute,threshold,simple_cell]=CART(features,labels,min_loss,min_num)
attribute=[];
threshold=[];
simple_cell=[];
loss=[];
[row,col]=size(features);
if row<3 || row<=min_num 
    return;
end
for i=1:col
    simple=[features,labels];
    simple=sortrows(simple,i);
    features=simple(:,1:end-1);
    labels=simple(:,end);
    cut_point=[];
    loss_j=[];
    simple_cell_temp=[];
    for j=1:row-1
        if(features(j,i)==features((j+1),i))
            continue;
        end
        cut_point_temp=(features(j,i)+features((j+1),i))/2;
        average1=sum(labels(1:j))/j;
        average2=sum(labels(j+1:end))/(row-j);
        loss_temp=sum((labels(1:j)-average1).^2)+sum((labels(j+1:end)-average2).^2);
        if loss_temp>=min_loss
            if isempty(cut_point)
                cut_point=cut_point_temp;
                loss_j=loss_temp;
                simple_cell_temp={features(1:j,:),labels(1:j,:);features(j+1:end,:),labels(j+1:end,:)};
            else
                if(loss_temp<loss_j)
                    cut_point=cut_point_temp;
                    loss_j=loss_temp;
                    simple_cell_temp={features(1:j,:),labels(1:j,:);features(j+1:end,:),labels(j+1:end,:)};
                end
            end
        end
    end
    if ~isempty(cut_point)
        if isempty(attribute)
            attribute=i;
            threshold=cut_point;
            loss=loss_j;
            simple_cell=simple_cell_temp;
        else
            if loss>loss_j
                attribute=i;
                threshold=cut_point;
                loss=loss_j;
                simple_cell=simple_cell_temp;
            end
        end
    end
end
end