function [newtestset] = redistribute(trainLabel)

% Element per Label
EpL = 200;

newtestset = zeros(1, EpL * 10);

[~,sumIndex] = sort(trainLabel,'ascend');
indexNum=zeros(10,1);

for ii=1:10
    indexNum(ii)=sum(length(find(trainLabel==ii)));
end
curIndex = 0;
for ii = 1:10
    newtestset(EpL*(ii-1)+1:EpL*ii) = sumIndex(curIndex + randi(indexNum(ii),1,EpL));
    curIndex = curIndex + indexNum(ii);
end

end
