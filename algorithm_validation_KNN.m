%% your classifer traing code here
function [valid_p] = algorithm_validation_KNN(x_train,y_train,x_test,EpL,cri,k,T)
    temp_p=zeros(length(x_test),T);
    for iter=1:T
        a = redistribute(y_train,EpL,cri);
        selected_y_train=y_train(a);
        [PCA_comp, PCA_score]=PCA_part(x_train(:,a));
        PCA_test=PCA_comp'*x_test;
        % Making output matrix
        for ii=1:length(PCA_test)
        % Iterations for all input points    
            d=sqrt(sum((PCA_test(:,ii)-PCA_score').^2));
            [~,I]=sort(d,'ascend');
            % Finding shortest distance points
            temp_p(ii,iter)=mode(selected_y_train(I(1:k)));
            % Designating classifier as most frequent label among k-nearest points
        end    
    end
    valid_p=mode(temp_p,2);
end