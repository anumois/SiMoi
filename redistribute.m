function [newtestset] = redistribute(trainLabel,EpL,cri)

% Element per Label
indexNum=zeros(cri,1);
newtestset = zeros(1, EpL * cri);

[~,sumIndex] = sort(trainLabel,'ascend');


for ii=1:length(indexNum)
    indexNum(ii)=sum(length(find(trainLabel==ii)));
end
curIndex = 0;
for ii = 1:length(indexNum)
    newtestset(EpL*(ii-1)+1:EpL*ii) = sumIndex(curIndex + randi(indexNum(ii),1,EpL));
    curIndex = curIndex + indexNum(ii);
end

end
