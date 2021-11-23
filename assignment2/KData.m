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