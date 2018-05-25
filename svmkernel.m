function K = svmkernel(x_i, x_j,sig)
%% Arguments %%
% x_i: ith random data from [m]
% x_j: j ~= i data from [m]
% kernel_type: kernel types : 'linear' or 'rbf'
%% Your code here %%

      %variance of kernel
      K=zeros(size(x_i,1),size(x_j,1));
for ii=1:size(x_i,1)      
      d=sqrt(sum((x_i(ii,:)'-x_j').^2));
      K(ii,:)=exp(-1*d/(2*(sig^2)));
end
      K = double(K);

end
% Convert to full matrix if inputs are sparse