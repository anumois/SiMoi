function conf_mat=confusion_generator(y_valid,valid_p)
   conf_mat=zeros(10);
   for ii=1:10
       for jj=1:10
        conf_mat(jj,ii)=sum(valid_p(y_valid==ii)==jj)/sum(y_valid==ii);
       end
   end
end