%% your classifer traing code here
function [model] = algorithm_kernelSVM_slow(comp,score, y_train,T,l,sig)
    tags=unique(y_train);
    model.alpha=zeros(length(y_train),9);
    model.structure=struct;
    model.structure(1).O=1:length(tags);
    PG_list=[]; NG_list=[];
    for ii=1:length(tags)-1
        for indx=1:2^(length(model.structure(ii).O)-1)-1
            p_n=zeros(length(model.structure(ii).O),1);
            n_n=zeros(length(model.structure(ii).O),1);
            cnt=indx;
            for jj=1:length(model.structure(ii).O)
                if rem(cnt,2)==1
                    p_n(jj)=1;
                else
                    n_n(jj)=1;
                end
                cnt=floor(cnt/2);
            end
            y_train_temp=zeros(length(y_train),1);
            for jj=1:length(n_n)
                if n_n(jj)==1
                    y_train_temp(y_train==model.structure(ii).O(jj))=-1;
                end
                if p_n(jj)==1
                    y_train_temp(y_train==model.structure(ii).O(jj))=1;
                end
            end      
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
            pred=svmkernel(score_temp,score_temp,sig)*sum(a(active_indx,:),2)/T;
            pred=1-((pred>0 & y_train_temp>0) | (pred<0 & y_train_temp<0));
            temp_err=sum(pred);
            if indx==1
                winner.a=sum(a,2)/T;
                winner.err=temp_err;
                winner.structure.P=model.structure(ii).O(boolean(p_n));
                winner.structure.N=model.structure(ii).O(boolean(n_n));
            elseif winner.err>temp_err
                winner.a=sum(a,2)/T;
                winner.err=temp_err;
                winner.structure.P=model.structure(ii).O(boolean(p_n));
                winner.structure.N=model.structure(ii).O(boolean(n_n));                
            end
        end    
        if length(winner.structure.P)>1
            PG_list=[PG_list, ii];
        end
        if length(winner.structure.N)>1
            NG_list=[NG_list, ii];
        end
        model.structure(ii).P=winner.structure.P;
        model.structure(ii).N=winner.structure.N;
        if isempty(PG_list)==0
            model.structure(PG_list(1)).PG=ii+1;
            model.structure(ii+1).O=model.structure(PG_list(1)).P;           
            PG_list=PG_list(2:end);
        elseif isempty(NG_list)==0
            model.structure(NG_list(1)).NG=ii+1;
            model.structure(ii+1).O=model.structure(NG_list(1)).N;           
            NG_list=NG_list(2:end);
        end
        model.alpha(:,ii)=winner.a;
    end
    % Averaging alpha 
    model.coeff=comp;
end