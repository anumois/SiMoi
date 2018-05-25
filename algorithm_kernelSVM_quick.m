%% your classifer traing code here
function [model] = algorithm_kernelSVM_quick(x_train, y_train,T,l,sig)
    [coeff,score,latent]=pca(x_train');
    latent_por=latent/sum(latent);
    coeff=coeff(:,1:find(latent_por<0.05,1));
    score=score(:,1:find(latent_por<0.05,1));
    model.alpha=zeros(length(y_train),9);
    model.structure=struct;
    model.structure(1).O=1:10;
%    set_chain=zeros(9,10);
    for ii=1:9
        disp(ii)
        temp_err=zeros(length(model.structure(ii).O),1);
        temp_a=zeros(length(y_train),length(model.structure(ii).O));
        indx=0;
        for c_n=model.structure(ii).O
            indx=indx+1;
%             y_train_temp=sum(y_train==find(set_chain(ii,:)==1),2)-sum(y_train==find(set_chain(ii,:)==-1),2);
            y_train_temp=zeros(length(y_train),1);
            if length(model.structure(ii).O)>=2
                y_train_temp(boolean(sum(y_train==model.structure(ii).O,2)))=-1;
            else
                y_train_temp(y_train==find(y_train==model.structure(ii).O))=-1;
            end
            y_train_temp(y_train==c_n)=1;
            active_indx=find(y_train_temp);
            y_train_temp=y_train_temp(active_indx);
            score_temp=score(active_indx,:);
            b=zeros(length(y_train),1); a=zeros(length(y_train),T);
            for t=1:T
                % Smaller gradient step size as iteration continues
                a(:,t)=b/(l*t);
                % Choosing random index from whole training data
                jj=ceil(length(y_train_temp)*rand());
                temp=y_train_temp(jj)*svmkernel(score_temp(jj,:),score_temp,sig)*a(active_indx,t);
                if temp<1
                    % Updting beta when prediction fails
                    b(active_indx(jj))=b(active_indx(jj))+y_train_temp(jj);
                end
            end
            temp_a(:,indx)=sum(a,2)/T;
            pred=svmkernel(score_temp,score_temp,sig)*temp_a(active_indx,indx);
            pred=1-((pred>0 & y_train_temp>0) | (pred<0 & y_train_temp<0));
            temp_err(indx)=sum(pred);
        end    
        temp_err=temp_err/length(active_indx);
        [~,I]=min(temp_err);
        model.structure(ii).P=model.structure(ii).O(I);
        model.structure(ii).N=model.structure(ii).O([1:I-1,I+1:end]);
        model.structure(ii).PG=[];
        model.structure(ii).NG=ii+1;
        model.structure(ii+1).O=model.structure(ii).N;
        model.alpha(:,ii)=temp_a(:,I);
    end
    % Averaging alpha 
    model.coeff=coeff;
end