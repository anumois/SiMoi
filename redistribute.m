function [newtestset] = redistribute(trainX,trainLabel)

% Element per Label
EpL = 200;

newtestset = zeros(1, EpL * 10);

index1 = [];
index2 = [];
index3 = [];
index4 = [];
index5 = [];
index6 = [];
index7 = [];
index8 = [];
index9 = [];
index10 = [];

for i = 1:length(trainLabel)
    switch trainLabel(i)
        case 10
            index10 = [i;index10];
        case 1
            index1 = [i;index1];
        case 2
            index2 = [i;index2];
        case 3
            index3 = [i;index3];
        case 4
            index4 = [i;index4];
        case 5
            index5 = [i;index5];    
        case 6
            index6 = [i;index6];
        case 7
            index7 = [i;index7];
        case 8
            index8 = [i;index8];
        case 9
            index9 = [i;index9];
    end
end


sumIndex = [index1;index2;index3;index4;index5;index6;index7;index8;index9;index10];
indexNum = [length(index1);length(index2);length(index3);length(index4);length(index5);length(index6);length(index7);length(index8);length(index9);length(index10);];

for i = 1:10
    curIndex = 0;
    for j = 1:EpL
        newtestset(EpL*(i-1) + j) = sumIndex(curIndex + randi(indexNum(i)));
    end
    curIndex = curIndex + indexNum(i);
end

end
