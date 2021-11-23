function entropy=getEntropy(labels,label1,label2)
count_all=length(labels);
count_lable1=length(find(labels==label1));
count_lable2=length(find(labels==label2));
if count_lable2==0
    entropy=0;
    return;
end
p1=count_lable1/count_all;
p2=count_lable2/count_all;
entropy=-(p1*log2(p1)+p2*log2(p2));
end