function [PCA_comp, PCA_score]=PCA_part(x_train)
    [coeff,score,latent]=pca(x_train');
    latent_por=latent/sum(latent);
    PCA_comp=coeff(:,1:find(latent_por<0.05,1));
    PCA_score=score(:,1:find(latent_por<0.05,1));
end