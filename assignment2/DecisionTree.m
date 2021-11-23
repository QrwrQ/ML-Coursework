function tree=DecisionTree(X,Y,varargin)
ip=inputParser;
ip.addParameter('corefunction',@(x)any(validatestring(x,{'ID3','CART'})));
ip.addOptional('layer',5,@(x)(x>1));
ip.addOptional('loss',0);
ip.addOptional('gain',0);
ip.addOptional('num',1,@(x)(x>=1));
ip.parse(varargin{:});
max_layer=ip.Results.layer;
min_num=ip.Results.num;
switch ip.Results.corefunction
    case "ID3"
        corefunc=@ID3;
        arg=ip.Results.gain;
    case "CART"
        corefunc=@CART;
        arg=ip.Results.loss;
end

tree=CreateTree_re(X,Y,corefunc,max_layer,arg,min_num,0);
end