
function [validp] = validation_NaiveBayes(model, x_valid, cri)    
    function [probVal] = getProb(model,feature,value, cri)
        [~,~,steps] = size(model.FeatureIndex);
        discVal = zeros(1,cri);
        probVal = zeros(1,cri);
        
        for outerIter = 1:cri
            for innerIter = 1:steps-1
                if innerIter == steps -1
                    discVal(outerIter) = steps;
                    break;
                end
                
                if (model.FeatureIndex(outerIter,feature,innerIter+1) == -1)
                    discVal(outerIter) = innerIter;
                    break;
                end
                
                if (value >= model.FeatureIndex(outerIter,feature,innerIter)) && (value < model.FeatureIndex(outerIter,feature,innerIter+1))
                    discVal(outerIter) = innerIter;
                    break;                
                    
                end
                
            end
        end
        
        for outerIter = 1:cri
            probVal(outerIter) = model.FeatureCount(outerIter,feature,discVal(outerIter)) / sum(model.FeatureCount(outerIter,feature,:));
        end
    end

    x_test = model.coeff' * x_valid;
    [numFeature, numValid] = size(x_test);
    
    validp = zeros(1,numValid);
    
    for ii = 1:numValid
        predictProb = ones(1,cri);
        for jj = 1:numFeature
            probVal = getProb(model, jj, x_test(jj,ii), cri);
            predictProb = predictProb .* probVal;
        end
        tmp = find(predictProb == max(predictProb));
        validp(ii) = tmp(1);
    end
    validp = transpose(validp);
end

