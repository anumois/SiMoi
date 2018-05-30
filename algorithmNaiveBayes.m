function [model] = algorithmNaiveBayes(x_train,y_train, cri, comp)    
    [numFeature, ~] = size(x_train);
    indexStore = round(cri * 100);
    model.FeatureIndex = zeros(cri,numFeature,indexStore);
    model.FeatureCount = zeros(cri,numFeature,indexStore);
    for ii = 1:cri
        labelSpecificX = x_train(:,y_train == cri);
        for jj = 1:numFeature
            uniqueFeatures = unique(labelSpecificX(jj,:));
            model.FeatureIndex(ii,jj,:) = [uniqueFeatures, -ones(1, indexStore - length(uniqueFeatures))];
            tmp = model.FeatureIndex(ii,jj,:);
            model.FeatureCount(ii,jj,:) = [sum(uniqueFeatures == labelSpecificX(:)) , -ones(1,indexStore - length(uniqueFeatures))];
        end
    end
    model.coeff = comp;
        
end

