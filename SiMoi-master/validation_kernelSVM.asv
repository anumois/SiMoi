%% validation code here
function [valid_p] = validation_kernelSVM(model, valid_feat, x_train,sig)
x_test=model.coeff'*valid_feat;
x_train=model.coeff'*x_train;
% Evaluate x_test
num_test = length(x_test);
valid_p = zeros(length(x_test),1);


    % Estimate feature map
    temp=svmkernel(x_test',x_train',sig)*model.alpha;
for ii=1:length(x_test)
    jj=1;
    while true
        if temp(ii,jj)>0
            if length(model.structure(jj).P)==1
                valid_p(ii)=model.structure(jj).P;
                break;
            else
                jj=model.structure(jj).PG;
            end
        else
            if length(model.structure(jj).N)==1
                valid_p(ii)=model.structure(jj).N;
                break;
            else
                jj=model.structure(jj).NG;
            end
        end
    end
end


end